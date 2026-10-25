"""Run a command in a throwaway Docker container to show its real output without touching the host."""
import os
import re
import secrets
import shlex
import shutil
import subprocess
from dataclasses import dataclass

IMAGE = os.environ.get("SHELLSAFE_SANDBOX_IMAGE", "ubuntu:24.04")


@dataclass
class RunResult:
    ok: bool
    exit_code: int | None
    output: str
    note: str = ""


def available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0


def _argv(workdir: str, script: str) -> list[str]:
    return [
        "docker", "run", "--rm", "--quiet", "--network", "none", "--memory", "256m", "--pids-limit", "64",
        "--cpus", "1", "--read-only", "--tmpfs", "/tmp:size=64m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "-v", f"{workdir}:/work:ro", "-w", "/work",
        # As the host user: root without capabilities cannot read owner-only folders (e.g. a 700 home).
        "--user", f"{os.getuid()}:{os.getgid()}",
        IMAGE, "bash", "-c", script,
    ]


def run(command: str, workdir: str, timeout: int = 10, max_chars: int = 2000) -> RunResult:
    """The current directory is mounted read-only, so the command sees real files but cannot change them."""
    argv = _argv(workdir, command)
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 5)
    except subprocess.TimeoutExpired:
        return RunResult(False, None, "", f"timed out after {timeout}s")
    text = (out.stdout + out.stderr)[:max_chars]
    note = ""
    if "Read-only file system" in text:
        note = "the sandbox is read-only, so commands that write files only show what they would attempt"
    elif out.returncode == 127:
        note = f"the command is not installed in the sandbox image ({IMAGE})"
    return RunResult(out.returncode == 0, out.returncode, text, note)


def run_many(commands: list[str], workdir: str, timeout: int = 3, max_chars: int = 2000) -> list[RunResult]:
    """Run several candidates one after another in a single container (one startup instead of one each).

    Each gets its own `timeout`, no stdin, and the same read-only view of `workdir`; output is split on a random
    marker the commands cannot predict.
    """
    mark = secrets.token_hex(8)
    script = "".join(f"timeout -k 1 {timeout} bash -c {shlex.quote(c)} </dev/null 2>&1; printf '\\n{mark} %d\\n' $?\n"
                     for c in commands)
    try:
        out = subprocess.run(_argv(workdir, script), capture_output=True, text=True,
                             timeout=(timeout + 1) * len(commands) + 10)
        text = out.stdout
    except subprocess.TimeoutExpired as e:
        text = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    parts = re.split(rf"\n{mark} (\d+)\n", text)
    results = []
    for i in range(len(commands)):
        if 2 * i + 1 >= len(parts):
            results.append(RunResult(False, None, "", "did not finish in the sandbox"))
            continue
        code = int(parts[2 * i + 1])
        note = f"timed out after {timeout}s" if code in (124, 137) else ""
        results.append(RunResult(code == 0, None if note else code, parts[2 * i][:max_chars], note))
    return results
