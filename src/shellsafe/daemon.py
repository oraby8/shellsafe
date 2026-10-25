"""Optional background process that keeps the model loaded, so `shellsafe ask` skips the ~3s model load.

Off unless the user starts it (`shellsafe daemon start`). `ask` uses it while it runs and loads the model itself
otherwise (or always, with --no-daemon). It listens on a Unix socket that only the user can open and answers one
request at a time.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from shellsafe.engine import Suggestion
from shellsafe.retrieve import Page
from shellsafe.safety import Risk, Verdict

HOME = Path(os.environ.get("SHELLSAFE_HOME", Path.home() / ".local/share/shellsafe"))
SOCK = HOME / "daemon.sock"
LOG = HOME / "daemon.log"


def to_wire(suggestions: list[Suggestion], pages: list[Page]) -> dict:
    return {"suggestions": [{"command": s.command, "votes": s.votes, "risk": int(s.verdict.risk),
                             "reasons": s.verdict.reasons, "flags": s.flags, "warnings": s.warnings}
                            for s in suggestions],
            "pages": [{"name": p.name, "description": p.description, "examples": p.examples} for p in pages]}


def from_wire(d: dict) -> tuple[list[Suggestion], list[Page]]:
    suggestions = [Suggestion(s["command"], s["votes"], Verdict(Risk(s["risk"]), s["reasons"]),
                              [tuple(f) for f in s["flags"]], s["warnings"]) for s in d["suggestions"]]
    pages = [Page(p["name"], p["description"], [tuple(e) for e in p["examples"]]) for p in d["pages"]]
    return suggestions, pages


def call(msg: dict, timeout: float = 120) -> dict | None:
    """Send one request; None when no daemon is listening."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(str(SOCK))
            s.sendall(json.dumps(msg).encode() + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                data += chunk
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError):
        return None
    return json.loads(data) if data else None


def status() -> dict | None:
    return call({"op": "status"}, timeout=5)


def start(model_path: str, wait: float = 120) -> dict:
    """Launch the daemon in its own session and wait until it has loaded the model."""
    if (st := status()) is not None:
        return st
    HOME.mkdir(parents=True, exist_ok=True, mode=0o700)
    SOCK.unlink(missing_ok=True)  # left over from a daemon that was killed
    subprocess.Popen([sys.executable, "-m", "shellsafe.daemon", model_path], stdin=subprocess.DEVNULL,
                     stdout=open(LOG, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        if (st := status()) is not None:
            return st
        time.sleep(0.3)
    raise RuntimeError(f"the daemon did not start; see {LOG}")


def stop() -> bool:
    return call({"op": "stop"}, timeout=5) is not None


def serve(model_path: str):
    from shellsafe import docs
    from shellsafe.engine import Suggester

    t0 = time.time()
    suggester = Suggester(model_path)
    learned_mtime = docs.LEARNED_DIR.stat().st_mtime if docs.LEARNED_DIR.exists() else 0
    HOME.mkdir(parents=True, exist_ok=True, mode=0o700)
    SOCK.unlink(missing_ok=True)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)  # socket created as 0600
    srv.bind(str(SOCK))
    os.umask(old)
    srv.listen(8)
    print(f"[{time.strftime('%F %T')}] ready: {model_path} (pid {os.getpid()}, load {time.time() - t0:.1f}s)",
          flush=True)
    try:
        while True:
            conn, _ = srv.accept()
            with conn:
                try:
                    data = b""
                    while not data.endswith(b"\n"):
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    msg = json.loads(data)
                    if msg["op"] == "stop":
                        conn.sendall(b'{"stopped": true}\n')
                        break
                    if msg["op"] == "status":
                        reply = {"model": model_path, "pid": os.getpid(), "uptime_s": round(time.time() - t0)}
                    else:
                        # `shellsafe learn` adds docs while the daemon runs: reload them when the folder changes.
                        mtime = docs.LEARNED_DIR.stat().st_mtime if docs.LEARNED_DIR.exists() else 0
                        if mtime != learned_mtime:
                            suggester.reload_docs()
                            learned_mtime = mtime
                        reply = to_wire(*suggester.suggest(msg["request"], n=msg["n"], samples=msg["samples"],
                                                           workdir=msg["workdir"]))
                    conn.sendall(json.dumps(reply).encode() + b"\n")
                except Exception as e:  # one bad request must not take the daemon down
                    print(f"[{time.strftime('%F %T')}] error: {type(e).__name__}: {e}", flush=True)
                    try:
                        conn.sendall(json.dumps({"error": f"{type(e).__name__}: {e}"}).encode() + b"\n")
                    except OSError:
                        pass
    finally:
        srv.close()
        SOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    serve(sys.argv[1])
