"""Runs inside the sandbox. Reads {"id","cmd"} JSON lines on stdin, writes one verdict per line on stdout."""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

PRISTINE = "/fixture-pristine"
TREES = ["home/user", "testbed", "system", "workspace"]
# No RLIMIT_NPROC: it counts uid 1001 across every container on the host, so parallel shards starved each other
# of forks. The container's cgroup pids limit bounds fork bombs, and cleanup below needs no fork.
AS_USER = ["setpriv", "--reuid=1001", "--regid=1001", "--init-groups", "--reset-env"]
BAD_USAGE = re.compile(
    r"invalid option|unrecognized option|unknown option|illegal option|invalid argument|"
    r"option requires an argument|unrecognized argument|unknown argument|no such option|"
    r"invalid (?:number|time|date|predicate|expression)|unknown predicate|paths must precede expression|"
    r"Try '[^']+ --help'|^usage:|syntax error|unexpected (?:token|EOF)", re.I | re.M)
NOT_FOUND = re.compile(r"(?:^|: )([\w.+-]+): (?:command )?not found", re.M)


def kill_user_processes():
    """kill(-1) as root signals every process in this PID namespace except init and the worker itself, in one
    syscall and without forking, so it works even when a fork bomb has filled the pids limit."""
    for _ in range(3):
        try:
            os.kill(-1, signal.SIGKILL)
        except ProcessLookupError:
            break
        time.sleep(0.02)


def restore():
    kill_user_processes()
    for t in TREES:
        dst = "/" + t
        subprocess.run(["rsync", "-a", "--delete", f"{PRISTINE}/{t}/", dst + "/"], check=True)


def classify(cmd: str) -> dict:
    syntax = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True)
    if syntax.returncode != 0:
        return {"status": "syntax", "stderr": syntax.stderr[-300:]}
    run_cmd = re.sub(r"(^|[|;&]\s*)sudo\s+(-\S+\s+)*", r"\1", cmd)  # verify without privilege escalation
    env = ["env", "HOME=/home/user", "USER=user", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LANG=C.UTF-8", "TERM=dumb"]
    try:
        p = subprocess.run(AS_USER + env + ["timeout", "-k", "1", "5", "bash", "-c", run_cmd],
                           capture_output=True, text=True, cwd="/home/user", stdin=subprocess.DEVNULL,
                           timeout=10, errors="replace")
        code, out, err = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        code, out, err = 124, "", ""
    missing = NOT_FOUND.search(err)
    if code == 127 or missing:
        status = "missing_tool"
    elif BAD_USAGE.search(err):
        status = "bad_usage"
    elif code == 124:
        status = "timeout"
    elif code == 0:
        status = "ok"
    else:
        status = "runtime_error"
    res = {"status": status, "exit": code, "stdout": out[:400], "stderr": err[-300:]}
    if missing:
        res["tool"] = missing.group(1)
    return res


def main():
    restore()
    for line in sys.stdin:
        item = json.loads(line)
        try:
            res = classify(item["cmd"])
        except Exception as e:  # never let one command stop the shard
            res = {"status": "worker_error", "stderr": str(e)[:300]}
        res["id"] = item["id"]
        print(json.dumps(res), flush=True)
        try:
            restore()
        except subprocess.CalledProcessError:
            pass


if __name__ == "__main__":
    main()
