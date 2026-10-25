"""Score generated commands with InterCode-ALFA (execution in Docker + local LLM judge).

Usage: python eval/score.py eval/results/<model>-<quant>.jsonl [--judge qwen3.8:latest]
Writes <input>.scored.jsonl and prints accuracy overall and by difficulty.
Verdicts are cached in eval/judge_cache.jsonl per (judge, task, command): the judge still flips ~2% of verdicts
between runs even at temperature 0, so the cache makes identical answers score identically across models and runs.
Seed it from existing scored runs with: python eval/score.py --seed-cache eval/det/*.scored.jsonl
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import requests
from icalfa import submit_command

CACHE = Path(__file__).parent / "judge_cache.jsonl"


def cache_key(judge, index, cmd):
    return f"{judge}|{index}|{' '.join(cmd.split())}"


def load_cache():
    cache = {}
    if CACHE.exists():
        for e in map(json.loads, CACHE.open()):
            cache.setdefault(e["key"], e["score"])  # first verdict wins
    return cache


def seed_cache(paths, judge):
    cache = load_cache()
    added = 0
    with CACHE.open("a") as f:
        for p in paths:
            for r in map(json.loads, open(p)):
                key = cache_key(judge, r["index"], r["cmd"])
                if r["cmd"] and key not in cache:
                    cache[key] = r["score"]
                    f.write(json.dumps({"key": key, "score": r["score"]}) + "\n")
                    added += 1
    print(f"cache: {added} added, {len(cache)} total")


def patch_judge(think: bool):
    """Fix icalfa's Ollama request: it sends temperature/seed at the top level, where Ollama ignores them, so the
    judge sampled at its default temperature (identical commands could score differently). Move them into
    `options`. Also turn thinking off unless asked (~1s per judgement instead of ~10s)."""
    original = requests.post

    def post(url, data=None, **kwargs):
        if url.endswith("/api/chat") and data:
            payload = json.loads(data)
            payload["options"] = {"temperature": 0, "seed": payload.pop("seed", 123)}
            payload.pop("temperature", None)
            if not think:
                payload["think"] = False
            data = json.dumps(payload)
        return original(url, data=data, **kwargs)

    requests.post = post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--judge", default="qwen3.8:latest")
    ap.add_argument("--judge-think", action="store_true", help="let the judge reason first (baseline setting, ~10x slower)")
    ap.add_argument("--out", help="scored output path (default: <input>.scored.jsonl)")
    ap.add_argument("--seed-cache", action="store_true", help="add the verdicts of already-scored files to the cache")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if args.seed_cache:
        seed_cache(args.results, args.judge)
        return
    patch_judge(think=args.judge_think)
    cache = {} if args.no_cache else load_cache()
    cache_f = None if args.no_cache else CACHE.open("a")

    src = Path(args.results[0])
    dst = Path(args.out) if args.out else src.with_suffix(".scored.jsonl")
    done = {}
    if dst.exists():
        done = {r["index"]: r for r in map(json.loads, dst.open())}

    rows = [json.loads(l) for l in src.open()]
    with dst.open("a") as f:
        for r in rows:
            if r["index"] in done:
                continue
            key = cache_key(args.judge, r["index"], r["cmd"])
            if not r["cmd"]:
                r["score"] = 0
            elif key in cache:
                r["score"] = cache[key]
            else:
                r["score"] = submit_command(r["index"], r["cmd"], eval_mode="ollama", eval_param=args.judge)
                cache[key] = r["score"]
                if cache_f:
                    cache_f.write(json.dumps({"key": key, "score": r["score"]}) + "\n")
                    cache_f.flush()
            f.write(json.dumps(r) + "\n")
            f.flush()
            done[r["index"]] = r

    by_diff = defaultdict(list)
    for r in done.values():
        by_diff[r["difficulty"]].append(r["score"])
    total = [s for v in by_diff.values() for s in v]
    lat = sorted(r["latency_s"] for r in done.values())
    print(f"{src.stem}: acc={sum(total)/len(total):.3f} n={len(total)} "
          + " ".join(f"d{d}={sum(v)/len(v):.2f}" for d, v in sorted(by_diff.items()))
          + f" p50={lat[len(lat)//2]:.2f}s p95={lat[int(len(lat)*.95)]:.2f}s")


if __name__ == "__main__":
    main()
