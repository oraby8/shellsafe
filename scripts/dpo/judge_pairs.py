"""Judge sampled commands that run but differ from gold, with the API models, and turn wrong ones into DPO pairs.

Usage:
  python scripts/dpo/judge_pairs.py judge SAMPLES.jsonl EXEC_OUT.jsonl JUDGED.jsonl [--limit 40000] [--concurrency 48]
  python scripts/dpo/judge_pairs.py build SAMPLES.jsonl JUDGED.jsonl EXEC_PAIRS_DIR OUT_DIR

judge: picks up to 2 samples per request that succeed or fail at runtime (not already rejected by pairs.py) and
whose output is not identical to the gold command's. One model judges each against the gold command and both
sandbox outputs; an "incorrect" verdict needs a second model to agree. Resumable by sample id.
build: merges the agreed-incorrect samples with the execution pairs (at most 2 rejected per request, judged ones
first since they cover the common failure) and writes OUT_DIR/{train,dev}.jsonl with the same dev requests.
Only training-split requests are ever sent to the API.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT / "scripts/data"))
sys.path.insert(0, str(_ROOT / "scripts/dpo"))
from pairs import reject_reason  # noqa: E402
from relabel import PRIMARY_WEIGHTS, SEMS, client, least_busy  # noqa: E402

MAX_PER_PROMPT = 2
SYSTEM = (
    "You grade a command suggested for a user's request; the target is ONE line of bash on Ubuntu Linux. You get a "
    "reference command known to be correct, and both commands' output from a synthetic sandbox (its files, users "
    "and services are made up, so errors about missing files, hosts or permissions are not the command's fault). "
    "The candidate is CORRECT if it does what the request asks: different tools, flags or order that achieve the "
    "same thing are fine, and so are different example values. It is INCORRECT if it does something else, misses "
    "part of the request, uses a wrong or invented flag, or would give the user the wrong result. Judge the command "
    "against the request, using the outputs as evidence. Reply with JSON only: {\"verdict\": \"correct\" or "
    "\"incorrect\"}")


def clip(v):
    out = (v.get("stdout") or "").strip()[:300]
    err = (v.get("stderr") or "").strip()[-200:]
    return f"exit {v.get('exit')}\nstdout: {out or '(empty)'}" + (f"\nstderr: {err}" if err else "")


async def ask(model, msg):
    async with SEMS[model]:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=0, max_completion_tokens=4000,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}])
                text = (r.choices[0].message.content or "").strip()
                if not text and r.choices[0].finish_reason == "length":
                    return "error"
                v = json.loads(text[text.find("{"):text.rfind("}") + 1]).get("verdict")
                if v in ("correct", "incorrect"):
                    return v
            except Exception:
                await asyncio.sleep(5 * 2 ** attempt)
        return "error"


def candidates(samples_path, exec_path, limit):
    res = {r["id"]: r["verify"] for r in map(json.loads, open(exec_path))}
    rng = random.Random(1)
    rows = []
    for r in map(json.loads, open(samples_path)):
        g = res.get(r["id"] + "|g")
        if not g or g["status"] in ("syntax", "bad_usage", "crashed_sandbox", "worker_error"):
            continue
        picks = []
        for i, s in enumerate(r["samples"]):
            v = res.get(f"{r['id']}|s{i}")
            if not v or v["status"] not in ("ok", "runtime_error") or reject_reason(g, v):
                continue
            if v["status"] == g["status"] == "ok" and v["stdout"] == g["stdout"] and g["stdout"].strip():
                continue  # same non-empty output as gold: a correct alternative
            picks.append((i, s, v))
        rng.shuffle(picks)
        for i, s, v in picks[:MAX_PER_PROMPT]:
            rows.append({"id": f"{r['id']}|s{i}", "request": r["messages"][-1]["content"], "gold": r["gold"],
                         "gold_out": clip(g), "cmd": s, "cmd_out": clip(v)})
    rng.shuffle(rows)
    return rows[:limit]


async def judge(args):
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    rows = [r for r in candidates(args.samples, args.exec_out, args.limit) if r["id"] not in done]
    print(f"{len(done)} judged, {len(rows)} to go")
    for m in PRIMARY_WEIGHTS:
        SEMS[m] = asyncio.Semaphore(args.concurrency)
    rng = random.Random(0)
    firsts = rng.choices(list(PRIMARY_WEIGHTS), weights=list(PRIMARY_WEIGHTS.values()), k=len(rows))
    bar = tqdm(total=len(rows), mininterval=30)
    with out.open("a") as f:
        async def one(r, first):
            msg = (f"{r['request']}\n\nReference: {r['gold']}\n{r['gold_out']}\n\n"
                   f"Candidate: {r['cmd']}\n{r['cmd_out']}")
            votes = {first: await ask(first, msg)}
            if votes[first] == "error":
                first = least_busy(exclude=first)
                votes[first] = await ask(first, msg)
            if votes[first] == "incorrect":
                second = least_busy(exclude=first)
                votes[second] = await ask(second, msg)
            f.write(json.dumps({**r, "votes": votes}) + "\n")
            f.flush()
            bar.update(1)
        await asyncio.gather(*(one(r, m) for r, m in zip(rows, firsts)))
    bar.close()


def build(args):
    by_prompt = {}
    stats = {"incorrect": 0, "correct": 0, "split": 0}
    for r in map(json.loads, open(args.judged)):
        v = list(r["votes"].values())
        if v.count("incorrect") >= 2:
            stats["incorrect"] += 1
            by_prompt.setdefault(r["id"].rsplit("|", 1)[0], []).append(r["cmd"])
        elif "correct" in v and "incorrect" in v:
            stats["split"] += 1
        else:
            stats["correct"] += 1
    print(stats)
    samples = {r["id"]: r for r in map(json.loads, open(args.samples))}
    exec_pairs = {}
    dev_ids = set()
    for split in ("train", "dev"):
        for p in map(json.loads, open(Path(args.exec_pairs) / f"{split}.jsonl")):
            exec_pairs.setdefault(p["id"], []).append(p["rejected"][0]["content"])
            if split == "dev":
                dev_ids.add(p["id"])
    rng = random.Random(0)
    new_ids = sorted(set(by_prompt) - set(exec_pairs))
    rng.shuffle(new_ids)
    dev_ids |= set(new_ids[:int(len(new_ids) * 0.03)])
    pairs = []
    for pid in set(by_prompt) | set(exec_pairs):
        r = samples[pid]
        for bad in (by_prompt.get(pid, []) + exec_pairs.get(pid, []))[:MAX_PER_PROMPT]:
            pairs.append({"id": pid, "source": r["source"], "prompt": r["messages"],
                          "chosen": [{"role": "assistant", "content": r["gold"]}],
                          "rejected": [{"role": "assistant", "content": bad}]})
    rng.shuffle(pairs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split, keep in (("train", lambda p: p["id"] not in dev_ids), ("dev", lambda p: p["id"] in dev_ids)):
        rows = [p for p in pairs if keep(p)]
        with (out / f"{split}.jsonl").open("w") as f:
            f.writelines(json.dumps(p) + "\n" for p in rows)
        print(split, len(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("samples")
    j.add_argument("exec_out")
    j.add_argument("out")
    j.add_argument("--limit", type=int, default=40000)
    j.add_argument("--concurrency", type=int, default=48, help="parallel requests per model (shared gateway)")
    b = sub.add_parser("build")
    b.add_argument("samples")
    b.add_argument("judged")
    b.add_argument("exec_pairs")
    b.add_argument("out_dir")
    args = ap.parse_args()
    if args.cmd == "judge":
        asyncio.run(judge(args))
    else:
        build(args)


if __name__ == "__main__":
    main()
