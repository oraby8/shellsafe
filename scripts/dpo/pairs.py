"""Build DPO preference pairs from sandbox-executed samples.

Usage:
  python scripts/dpo/pairs.py exec SAMPLES.jsonl EXEC_IN.jsonl        # rows for scripts/data/verify.py
  python scripts/dpo/pairs.py build SAMPLES.jsonl EXEC_OUT.jsonl OUT_DIR

The gold command is re-run next to the samples so both see the same fixture. Chosen is always the gold command.
A sample is rejected only on clear evidence that it is wrong:
  broken   - syntax error, bad flags/usage, or it crashed the sandbox
  invented - its tool is not installed and has no tldr page (common/linux), while the gold command's tool runs
  failed   - gold succeeds with output, the sample exits non-zero (timeouts are skipped: heavy scans are noisy)
A sample that succeeds is never rejected, even with different or no output: it may be a valid variant
(e.g. `-delete` instead of `xargs rm -v`), and a judge would be needed to tell.
Writes OUT_DIR/{train,dev}.jsonl in TRL conversational preference format.
"""
import collections
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT / "src"))
from shellsafe.retrieve import PLATFORMS, TLDR_ROOT  # noqa: E402

KNOWN_TOOLS = {p.stem for d in PLATFORMS for p in (TLDR_ROOT / d).glob("*.md")}

MAX_PER_PROMPT = 2
DEV_FRACTION = 0.03


def exec_rows(samples_path, out_path):
    with open(out_path, "w") as f:
        for r in map(json.loads, open(samples_path)):
            if not r["samples"]:
                continue
            f.write(json.dumps({"id": r["id"] + "|g", "cmd": r["gold"]}) + "\n")
            for i, s in enumerate(r["samples"]):
                f.write(json.dumps({"id": f"{r['id']}|s{i}", "cmd": s}) + "\n")


def reject_reason(g, s):
    gs, ss = g["status"], s["status"]
    if ss in ("syntax", "bad_usage", "crashed_sandbox"):
        return "broken"
    if ss == "missing_tool" and gs != "missing_tool" and s.get("tool") not in KNOWN_TOOLS:
        return "invented"
    if gs == "ok" and g.get("stdout", "").strip() and ss == "runtime_error":
        return "failed"
    return None


def build(samples_path, exec_path, out_dir):
    res = {}
    for r in map(json.loads, open(exec_path)):
        res[r["id"]] = r["verify"]
    stats = collections.Counter()
    pairs = []
    rng = random.Random(0)
    for r in map(json.loads, open(samples_path)):
        g = res.get(r["id"] + "|g")
        if not r["samples"] or g is None:
            continue
        if g["status"] in ("syntax", "bad_usage", "crashed_sandbox", "worker_error"):
            stats["skip:gold_broken"] += 1
            continue
        bad = []
        for i, s in enumerate(r["samples"]):
            v = res.get(f"{r['id']}|s{i}")
            why = v and reject_reason(g, v)
            stats[why or "not_rejected"] += 1
            if why:
                bad.append((why, s))
        rng.shuffle(bad)
        # Prefer different failure kinds when there are more than we keep.
        first = {why: s for why, s in reversed(bad)}
        kept = list(first.values()) + [s for _, s in bad if s not in first.values()]
        for s in kept[:MAX_PER_PROMPT]:
            pairs.append({"id": r["id"], "source": r["source"], "prompt": r["messages"],
                          "chosen": [{"role": "assistant", "content": r["gold"]}],
                          "rejected": [{"role": "assistant", "content": s}]})
    rng.shuffle(pairs)
    ids = sorted({p["id"] for p in pairs})
    rng.shuffle(ids)
    dev_ids = set(ids[:int(len(ids) * DEV_FRACTION)])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, keep in (("train", lambda p: p["id"] not in dev_ids), ("dev", lambda p: p["id"] in dev_ids)):
        rows = [p for p in pairs if keep(p)]
        with (out / f"{split}.jsonl").open("w") as f:
            f.writelines(json.dumps(p) + "\n" for p in rows)
        print(split, len(rows))
    print(dict(stats))
    print("prompts with pairs", len(ids))


if __name__ == "__main__":
    if sys.argv[1] == "exec":
        exec_rows(sys.argv[2], sys.argv[3])
    else:
        build(sys.argv[2], sys.argv[3], sys.argv[4])
