"""shellsafe: describe what you want, get a shell command with explanation, risk rating and real output."""
import os
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from shellsafe import docs, sandbox
from shellsafe.safety import Risk, assess

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()

RISK_STYLE = {Risk.SAFE: ("green", "safe"), Risk.CAUTION: ("yellow", "caution"), Risk.DANGEROUS: ("red", "DANGEROUS")}


def _risk_line(verdict) -> str:
    color, label = RISK_STYLE[verdict.risk]
    reasons = f" — {'; '.join(verdict.reasons)}" if verdict.reasons else ""
    return f"[{color}]● {label}[/{color}]{reasons}"


@app.command()
def ask(
    request: list[str] = typer.Argument(..., help="What you want to do, in plain words"),
    n: int = typer.Option(3, "-n", help="How many alternatives to show"),
    try_it: bool = typer.Option(False, "--try", help="Run the top safe command in a read-only Docker sandbox"),
    fast: bool = typer.Option(False, "--fast", help="One answer, no sampling or sandbox vote (~5x faster, less accurate)"),
    model: Optional[str] = typer.Option(None, help="Path to a GGUF model"),
    no_daemon: bool = typer.Option(False, "--no-daemon", help="Load the model in this process even if the daemon runs"),
):
    """Suggest commands for a free-text request."""
    from shellsafe import daemon

    text = " ".join(request)
    samples = 0 if fast else 5
    t0 = time.perf_counter()
    reply = None
    with console.status("thinking..."):
        if not no_daemon:
            st = daemon.status()
            if st and (model is None or os.path.abspath(model) == st["model"]):
                reply = daemon.call({"op": "ask", "request": text, "n": n, "samples": samples, "workdir": os.getcwd()})
        if reply is not None and "error" not in reply:
            suggestions, pages = daemon.from_wire(reply)
            timing = "via daemon"
        else:
            from shellsafe.engine import Suggester

            try:
                suggester = Suggester(model)
            except RuntimeError as e:  # the model could not be found or downloaded
                console.print(f"[red]{e}[/red]")
                raise typer.Exit(1)
            t_load = time.perf_counter() - t0
            suggestions, pages = suggester.suggest(text, n=n, samples=samples, workdir=os.getcwd())
            timing = f"model load {t_load:.2f}s"
            if reply is None and not no_daemon:
                timing += "; `shellsafe daemon start` keeps the model loaded between calls"
    elapsed = time.perf_counter() - t0
    if not suggestions:
        console.print("[red]No command found.[/red] Try rephrasing, or teach the tool with `shellsafe learn <tool>`.")
        raise typer.Exit(1)

    for i, s in enumerate(suggestions):
        title = "best" if i == 0 else f"alternative {i}"
        body = [f"[bold]{s.command}[/bold]", "", _risk_line(s.verdict)]
        if s.flags:
            body += [""] + [f"  [cyan]{f}[/cyan]  {d}" for f, d in s.flags]
        body += [f"  [yellow]![/yellow] {w}" for w in s.warnings]
        console.print(Panel("\n".join(body), title=title, title_align="left",
                            border_style=RISK_STYLE[s.verdict.risk][0]))

    from shellsafe.retrieve import command_names

    used = set(command_names(suggestions[0].command))
    examples = [e for p in pages if p.name in used for e in p.examples[:3]]
    if examples:
        console.print("[dim]Related examples:[/dim]")
        for what, cmd in examples:
            console.print(f"  [dim]{what}:[/dim] {cmd}")

    if try_it:
        _try(suggestions[0].command, suggestions[0].verdict)
    console.print(f"[dim]{elapsed:.2f}s ({timing}). Commands are never run on your machine.[/dim]")


def _try(command: str, verdict):
    if verdict.risk != Risk.SAFE:
        console.print("[yellow]Not running it in the sandbox: it is not rated safe.[/yellow]")
        return
    if not sandbox.available():
        console.print("[yellow]Docker is not available, so the expected output cannot be shown.[/yellow]")
        return
    with console.status("running in sandbox..."):
        r = sandbox.run(command, os.getcwd())
    title = f"expected output (sandbox, exit {r.exit_code})"
    console.print(Panel(r.output.rstrip() or "(no output)", title=title, title_align="left", border_style="blue"))
    if r.note:
        console.print(f"[dim]{r.note}[/dim]")


daemon_app = typer.Typer(help="Optional background process that keeps the model loaded (off unless you start it).")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("start")
def daemon_start(model: Optional[str] = typer.Option(None, help="Path to a GGUF model")):
    """Load the model once and keep it in memory; `ask` then answers without the load time."""
    from shellsafe import daemon
    from shellsafe.engine import default_model

    with console.status("loading the model..."):
        try:
            st = daemon.start(os.path.abspath(model or default_model()))
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    console.print(f"daemon running (pid {st['pid']}) with {st['model']}. Stop it with `shellsafe daemon stop`.")


@daemon_app.command("stop")
def daemon_stop():
    """Stop the daemon and free its memory."""
    from shellsafe import daemon

    console.print("daemon stopped." if daemon.stop() else "the daemon is not running.")


@daemon_app.command("status")
def daemon_status():
    """Show whether the daemon runs and which model it holds."""
    from shellsafe import daemon

    st = daemon.status()
    console.print(f"running (pid {st['pid']}, up {st['uptime_s']}s) with {st['model']}" if st
                  else "not running. Start it with `shellsafe daemon start`.")


@app.command()
def check(command: str = typer.Argument(..., help="A command to rate, in quotes")):
    """Rate how risky a command is, without running it."""
    console.print(_risk_line(assess(command)))


@app.command()
def learn(
    tool: str = typer.Argument(..., help="Command name to index"),
    docs_file: Optional[Path] = typer.Option(None, "--docs", help="Index this file (README, .md, help dump) instead of --help/man"),
):
    """Teach shellsafe a new tool from its --help/man page or a docs file."""
    try:
        page = docs.learn(tool, docs_file)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)
    from shellsafe.retrieve import parse_page

    p = parse_page(page)
    console.print(f"Learned [bold]{tool}[/bold]: {len(p.examples)} options indexed → {page}")


if __name__ == "__main__":
    app()
