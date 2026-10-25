"""Local documentation for commands: man pages, tldr pages, and tools the user taught with `shellsafe learn`."""
import os
import re
import shutil
import subprocess
from pathlib import Path

LEARNED_DIR = Path(os.environ.get("SHELLSAFE_HOME", Path.home() / ".local/share/shellsafe")) / "learned"


def man_text(name: str) -> str:
    if not shutil.which("man"):
        return ""
    try:
        out = subprocess.run(["man", "-P", "cat", name], capture_output=True, text=True, timeout=5,
                             env={**os.environ, "MANWIDTH": "200", "MAN_KEEP_FORMATTING": "0", "GROFF_NO_SGR": "1"})
    except subprocess.TimeoutExpired:
        return ""
    if out.returncode != 0:
        return ""
    # Strip backspace overstrike and ANSI escape formatting.
    return re.sub(r"\x1b\[[0-9;]*m", "", re.sub(r".\x08", "", out.stdout))


def help_text(name: str) -> str:
    """Run `<name> --help`. Only called from `learn`, i.e. when the user explicitly asks for that tool."""
    if not shutil.which(name):
        return ""
    for flag in ("--help", "-h", "help"):
        try:
            out = subprocess.run([name, flag], capture_output=True, text=True, timeout=5, stdin=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, OSError):
            continue
        text = out.stdout or out.stderr
        if len(text) > 80:
            return text
    return ""


OPTION_LINE = re.compile(r"^\s{0,8}((?:-{1,2}[\w][\w-]*(?:[ =]<?[\w.|-]+>?)?(?:,\s*)?)+)\s{2,}(\S.*)$")
# man-page style: the option alone on a line, its description indented further on the next lines.
OPTION_ALONE = re.compile(r"^(\s{0,10})(-{1,2}[\w][\w-]*(?:[ =,\[].{0,50})?)$")


def options_from_help(text: str) -> list[tuple[str, str]]:
    """Pull (flags, description) pairs out of --help or man text."""
    opts, last = [], None
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = OPTION_LINE.match(line)
        alone = OPTION_ALONE.match(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if m:
            last = [m.group(1).strip().rstrip(","), m.group(2).strip()]
            opts.append(last)
        elif alone and nxt.strip() and len(nxt) - len(nxt.lstrip()) > len(alone.group(1)):
            last = [alone.group(2).strip(), ""]
            opts.append(last)
        elif last and last[1] == "" and line.strip():
            last[1] = line.strip()
        elif last and line.strip() and line.startswith(" " * 10) and len(last[1]) < 200:
            last[1] += " " + line.strip()  # wrapped description
        else:
            last = None
    return [tuple(o) for o in opts]


def to_tldr_page(name: str, text: str, max_examples: int = 400) -> str:
    """Render help text as a tldr-style page so the normal lookup can index it."""
    first = next((l.strip() for l in text.splitlines()
                  if re.match(r"^\s*[A-Za-z]", l) and not l.lower().lstrip().startswith("usage")), name)
    lines = [f"# {name}", "", f"> {first[:200]}", ""]
    for flags, desc in options_from_help(text)[:max_examples]:
        flag = flags.split(",")[0].split()[0]
        lines += [f"- {desc.rstrip('.')}:", "", f"`{name} {flag}`", ""]
    return "\n".join(lines)


def learn(name: str, source: Path | None = None) -> Path:
    """Index a tool from its --help/man output, or from a docs file the user points at."""
    if source:
        text = source.read_text(encoding="utf-8", errors="replace")
    else:
        text = help_text(name) + "\n" + man_text(name)
    if not text.strip():
        raise ValueError(f"no --help, man page or docs found for {name!r}")
    LEARNED_DIR.mkdir(parents=True, exist_ok=True)
    (LEARNED_DIR / f"{name}.help.txt").write_text(text)
    page = LEARNED_DIR / f"{name}.md"
    page.write_text(text if source and source.suffix == ".md" and text.startswith("# ") else to_tldr_page(name, text))
    return page


def doc_text(name: str) -> str:
    """man page if installed, else the help text saved by `learn`."""
    learned = LEARNED_DIR / f"{name}.help.txt"
    return man_text(name) or (learned.read_text() if learned.exists() else "")


def flag_description(name: str, flag: str, doc: str) -> str | None:
    """Find what a flag does in a command's man/help text."""
    for flags, desc in options_from_help(doc):
        if re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", flags):
            return desc.split(". ")[0].rstrip(".")
    return None
