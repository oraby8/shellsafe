"""Targeted SFT data for v2's systematic benchmark errors, written by the local teacher (vLLM on :8100).

  python scripts/targeted/gen.py make   OUT.jsonl [EARLIER.jsonl] --calls 3000   # requests + commands grounded in the sandbox fixture
  python scripts/targeted/gen.py answer IN.jsonl OUT.jsonl       # the teacher answers each request blind
  python scripts/targeted/gen.py build  GEN_V.jsonl ANS_V.jsonl OUT_DIR   # keep pairs whose sandbox outputs agree

GEN_V / ANS_V are the make / answer outputs after scripts/data/verify.py. Error types targeted (from the v2 error
analysis): filenames treated as strings, /home/user/ prefixes on absolute dirs, commands that never exit, and
multi-condition find pipelines. No NL2SH-ALFA test item is shown to the teacher; near-duplicates of test requests
are dropped in build.
"""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT / "scripts/data"))
from teacher import PAIRS_SCHEMA, chat, run  # noqa: E402

SFT_SYSTEM = ("You are a bash command generator. Given a natural language request, output only the bash command "
              "that accomplishes it. No explanation, no markdown fences.")

FILES = [
    ("/testbed/textfile.txt", "5 lines of text, includes 'error: disk full' and 'warning: low memory'"),
    ("/testbed/hello.c", "small C program"), ("/testbed/hello.php", "one-line PHP script"),
    ("/testbed/FooBar.html", "HTML page with a heading and a link"), ("/testbed/dir1/subdir1/list.txt", "lines a, b, c"),
    ("/testbed/dir2/data.csv", "CSV with header name,age,city,salary and 4 rows"), ("/testbed/dir1", "folder"),
    ("/testbed", "folder with files and subfolders dir1, dir2"),
    ("/system/folder1/a.log", "one log line"), ("/system/folder1/b.txt", "one line"),
    ("/system/folder2/access.log", "50 web log lines like '2026-10-03 12:2:00 INFO request id=2 status=300 "
     "path=/api/v1/items'"), ("/system", "folder with folder1 and folder2"),
    ("/workspace/src/app.py", "python script with a TODO comment"), ("/workspace/src/util.py", "python add function"),
    ("/workspace", "folder with src/"),
    ("~/notes.txt", "5 lines of text (the current directory is ~ = /home/user)"), ("~/link_to_notes", "symlink"),
    ("~/.hidden_file", "hidden file"), ("~/Documents/readme.md", "4 lines, includes 'TODO: fix this'"),
    ("~/Documents/people.csv", "CSV name,age,city,salary"), ("~/logs/access.log", "50 web log lines with status="),
    ("~/logs/auth.log", "20 sshd 'Failed password for root from 10.0.0.N' lines"),
    ("~/.config/app/config.json", "JSON with users (name, active, roles) and version"),
    ("~/Downloads/big.bin", "3 MB binary"), ("~/Downloads/small.bin", "4 KB, modified 2026-01-15"),
    ("~/Pictures/red.png", "64x48 image"), ("~/Pictures/photo.jpg", "640x480 image"),
    ("~/backup/docs.tar.gz", "tarball"), ("~/backup/projects.zip", "zip"), ("~/backup/notes.txt.gz", "gzip"),
    ('"~/backup/notes copy.txt"', "name with a space, old"), ('"~/my folder/file with spaces.txt"', "spaces in path"),
    ("~/projects/webapp", "git repo with uncommitted change, src/*.py, src/main.c, index.html, .env"),
]
CATEGORIES = {
    "file_content": (25, "operations on the CONTENTS of a named file; the command reads the file itself and never "
                     "echoes its name or treats the name as a string",
                     ["count words", "count characters", "count bytes", "count lines", "print line N",
                      "print lines N to M", "last N lines", "first N lines", "md5 or sha256 checksum",
                      "sort lines", "unique lines", "reverse line order", "number the lines", "uppercase the text",
                      "count occurrences of a word", "lines not containing a word", "longest line",
                      "extract a column", "show non-empty lines", "file type", "word frequency",
                      "replace a word and print", "print every other line", "show size in bytes"]),
    "abs_dirs": (20, "tasks scoped to a named absolute folder (/testbed, /system, /workspace or their subfolders); "
                 "the command uses that exact absolute path, never /home/user or . in its place",
                 ["list contents with details", "count files recursively", "total disk usage",
                  "list files by extension", "show permissions", "newest file", "largest file", "list only folders",
                  "tree-like listing", "copy a file", "rename a file", "make a tar.gz of the folder",
                  "list files sorted by size", "count lines of all files", "search a word recursively",
                  "make all files read-only", "list hidden files", "show inode numbers", "create a subfolder"]),
    "find_complex": (20, "find with several conditions",
                     ["alternatives grouped with \\( -name A -o -name B \\)", "-maxdepth / -mindepth",
                      "-size thresholds", "-mtime / -newer", "-perm", "-empty", "-iname", "-not / -prune",
                      "-exec ... {} +", "piping to xargs", "count results with wc -l", "sort results by size",
                      "sort by modification time", "-type d vs -type f", "print with -printf"]),
    "noninteractive": (10, "system, process and network information that prints once and EXITS (no command that "
                       "waits for input or runs until interrupted)",
                       ["top once in batch mode", "ping a host a fixed number of times", "processes sorted by memory",
                        "processes sorted by CPU", "memory usage", "disk space", "uptime and load", "listening ports",
                        "IP addresses", "default route", "last journal lines without pager", "CPU details",
                        "logged-in users", "date in a format", "environment variables", "kernel version",
                        "process count", "open files of a process", "DNS lookup"]),
    "multistep": (20, "two or three operations combined in one pipeline",
                  ["count values of a field sorted by frequency", "sum a CSV column", "average a CSV column",
                   "filter CSV rows by a value", "top N failed-login IPs", "files containing a word with counts",
                   "extract and sort fields", "compare two files", "jq filter on JSON", "grep then count unique",
                   "find then archive", "find then total size", "find then grep inside", "lines matching two words",
                   "group by a column with awk", "paths from a log sorted by hits"]),
    "cwd_home": (5, "requests about the current directory, 'here' or the home folder; use . or ~ correctly and "
                 "quote names with spaces",
                 ["hidden files here", "follow a symlink", "file with spaces in its name", "sizes of folders here",
                  "files changed in the git repo", "count files here", "list here sorted by time"]),
}
STYLE = ("Mix terse imperative instructions and casual questions; vary wording and length. Each command is ONE line "
         "of bash for Ubuntu, real flags only, the simplest idiomatic command that fully does the request.")


def make_items(calls, prefix="tgt2", weights=None, hard_frac=0.5):
    rng = random.Random(prefix)
    weights = weights or {n: c[0] for n, c in CATEGORIES.items()}
    names = list(weights)
    weights = list(weights.values())
    items = []
    for i in range(calls):
        cat = rng.choices(names, weights)[0]
        items.append({"work_id": f"{prefix}:{i}", "cat": cat, "files": rng.sample(FILES, 5),
                      "ops": rng.sample(CATEGORIES[cat][2], min(4, len(CATEGORIES[cat][2]))), "hard": rng.random() < hard_frac,
                      "relative": rng.random() < 0.5})
    return items


async def make_worker(it):
    hard = "Make most of them harder (two or more conditions or steps)." if it.get("hard", it.get("seed", 0) > 0.5) else ""
    if it.get("relative"):  # the current directory is home, so bare names are how users usually refer to these
        files = "\n".join(f"- {p.replace('~/', '')}: {d}" for p, d in it["files"])
        home = "Name files in the current directory by their bare relative name (e.g. notes.txt), without ~/"
    else:
        files = "\n".join(f"- {p}: {d}" for p, d in it["files"])
        home = "Write home paths as ~/... or /home/user/..."
    msg = (f"Files and folders that exist on the machine:\n{files}\n\nCategory: {CATEGORIES[it['cat']][1]}.\n"
           f"Ideas to use: {'; '.join(it['ops'])}.\n\nWrite 6 different user requests in this category about the "
           f"files above, each with its best single-line command. Each request names the file or folder it acts on "
           f"({home}). {hard} {STYLE}")
    out = await chat([{"role": "user", "content": msg}], PAIRS_SCHEMA, temperature=1.0, max_tokens=1400)
    return [{"nl": p["request"].strip(), "cmd": p["command"].strip(), "cat": it["cat"]}
            for p in out["pairs"] if p["request"].strip() and p["command"].strip()]


async def answer_worker(it):
    out = await chat([{"role": "system", "content": SFT_SYSTEM}, {"role": "user", "content": f"Request: {it['nl']}"}],
                     {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
                     temperature=0, max_tokens=300)
    return [{"id": it["id"], "nl": it["nl"], "cmd": out["command"].strip()}]


MUTATING = re.compile(r"^(sudo )?(cp|mv|rm|mkdir|chmod|chown|touch|tar|zip|gzip|ln|rename|sed -i|truncate)\b")


def paths(cmd):
    return {t for t in re.findall(r"[\w./~-]*/[\w./-]+|\b[\w-]+\.\w{1,5}\b", cmd)}


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def build(gen_v, ans_v, out_dir):
    ans = {r["id"]: r for r in map(json.loads, open(ans_v))}
    test = [words(n) for n in pd.read_csv(ROOT / "data/raw/NL2SH-ALFA/test.csv").nl]
    seen, rows, why = set(), [], {}
    for g in map(json.loads, open(gen_v)):
        a = ans.get(g["id"])
        gv, av = g["verify"], (a or {}).get("verify", {})
        out_g, out_a = gv.get("stdout", "").strip(), av.get("stdout", "").strip()
        if gv["status"] != "ok":
            reason = "gen_" + gv["status"]
        elif not a or av.get("status") != "ok":
            reason = "ans_fail"
        elif out_g and out_g != out_a:
            reason = "disagree"
        elif not out_g and not MUTATING.search(g["cmd"]):
            reason = "empty_read"  # an empty result cannot show the command did the right thing
        elif not out_g and (out_a or g["cmd"].split()[0] != a["cmd"].split()[0] or paths(g["cmd"]) != paths(a["cmd"])):
            reason = "disagree_silent"
        elif "/home/user" in g["cmd"] and not re.search(r"/home/user|~|home", g["nl"]):
            reason = "home_prefix"
        elif g["nl"].lower() in seen:
            reason = "dup"
        elif any(len(words(g["nl"]) & t) / max(1, len(words(g["nl"]) | t)) >= 0.6 for t in test):
            reason = "near_test"
        else:
            reason = "keep"
        why[reason] = why.get(reason, 0) + 1
        if reason != "keep":
            continue
        seen.add(g["nl"].lower())
        cmd = min(g["cmd"], a["cmd"], key=len)  # both give the same result; teach the simpler one
        nl = g["nl"]
        if "~/" in nl and "~/" in cmd and int(g["id"].rsplit(":", 1)[1]) % 2:  # half: bare names, cwd is home
            nl, cmd = nl.replace("~/", ""), cmd.replace("~/", "")
        rows.append({"messages": [{"role": "system", "content": SFT_SYSTEM},
                                  {"role": "user", "content": f"Request: {nl}"},
                                  {"role": "assistant", "content": cmd}],
                     "id": g["id"], "source": f"targeted:{g['cat']}", "mode": "none"})
    print(json.dumps(why, indent=1))
    cats = {}
    for r in rows:
        cats[r["source"]] = cats.get(r["source"], 0) + 1
    print(json.dumps(cats, indent=1))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / "targeted.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "answer", "build"])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--calls", type=int, default=3000)
    ap.add_argument("--prefix", default="tgt2")
    ap.add_argument("--cats", help="category weights, e.g. find_complex=40,multistep=40")
    ap.add_argument("--hard-frac", type=float, default=0.5)
    args = ap.parse_args()
    if args.mode == "make":
        out = Path(args.paths[0])
        weights = dict((kv.split("=")[0], int(kv.split("=")[1])) for kv in args.cats.split(",")) if args.cats else None
        asyncio.run(run(make_items(args.calls, args.prefix, weights, args.hard_frac), make_worker, out))
        key = lambda r: " ".join(re.findall(r"[a-z0-9/._~-]+", r.get("nl", "").lower()))
        seen = {key(json.loads(l)) for path in args.paths[1:] for l in open(path)}  # earlier rounds
        with open(out.with_suffix(".ids.jsonl"), "w") as f:  # verify.py input, one row per new distinct request
            for i, r in enumerate(map(json.loads, open(out))):
                if r.get("empty") or key(r) in seen:
                    continue
                seen.add(key(r))
                f.write(json.dumps({"id": f"{r['work_id']}:{i}", **r}) + "\n")
    elif args.mode == "answer":
        items = [{"work_id": r["id"], **r} for r in map(json.loads, open(args.paths[0]))]
        asyncio.run(run(items, answer_worker, Path(args.paths[1])))
    else:
        build(*args.paths)


if __name__ == "__main__":
    main()
