"""How many wrong answers show a visible signal in the sandbox? Upper bound for a run-and-repair loop.

Usage (from eval/): python fail_signal.py qwen05-v45
Reads cached/<m>-vote.scored.jsonl and vote/<m>.cands.v.jsonl (sandbox results per distinct candidate).
"""
import json
import re
import sys
from collections import Counter

m = sys.argv[1]
MUTATING = re.compile(r"\b(rm|mv|cp|mkdir|rmdir|touch|chmod|chown|ln|tee|sed\s+-i|truncate|tar|zip|unzip|gzip|"
                      r"gunzip|dd|kill|pkill|useradd|apt|apt-get|pip|git\s+(commit|init|clone))\b|>")
run = {}
for line in open(f"vote/{m}.cands.v.jsonl"):
    d = json.loads(line)
    run[(d["id"].split("|")[0], d["cmd"])] = d["verify"]


def signal(cmd, v):
    if v is None:
        return "not_run"
    if v["status"] != "ok":
        return v["status"]  # runtime_error / bad_usage / missing_tool / syntax / timeout
    if not v["stdout"].strip():
        return "ok_silent_mutating" if MUTATING.search(cmd) else "ok_empty_output"
    return "ok_with_output"


SILENT = ("ok_with_output", "ok_silent_mutating", "not_run")
rows = [json.loads(line) for line in open(f"cached/{m}-vote.scored.jsonl")]
by = {0: Counter(), 1: Counter()}
diff = Counter()
examples = {}
for r in rows:
    s = signal(r["cmd"], run.get((str(r["index"]), r["cmd"])))
    by[r["score"]][s] += 1
    if r["score"] == 0:
        diff[(r["difficulty"], s not in SILENT)] += 1
        examples.setdefault(s, []).append((r["nl"][:70], r["cmd"][:70]))

wrong, right = sum(by[0].values()), sum(by[1].values())
print(f"{m}: {right} right, {wrong} wrong of {len(rows)}")
for name, c, tot in (("WRONG", by[0], wrong), ("RIGHT", by[1], right)):
    print(f"\n{name} answers by sandbox signal:")
    for s, k in c.most_common():
        print(f"  {s:20s} {k:4d}  {k / tot:5.1%}")
visible = lambda c: sum(k for s, k in c.items() if s not in SILENT)  # noqa: E731
print(f"\nwrong with a visible signal: {visible(by[0])}/{wrong} = {visible(by[0]) / wrong:.1%}")
print(f"right but with a signal (a repair turn could break them): {visible(by[1])}/{right} = {visible(by[1]) / right:.1%}")
print("\nwrong, by difficulty (signal / no signal):")
for d in (0, 1, 2):
    print(f"  d{d}: {diff[(d, True)]} / {diff[(d, False)]}")
for s, ex in examples.items():
    print(f"\n-- {s}")
    for nl, c in ex[:4]:
        print(f"   {nl!r} -> {c!r}")
