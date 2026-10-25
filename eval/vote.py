"""Self-consistency: sample several commands per test prompt, run them in the sandbox, answer with the majority.

  python eval/vote.py sample <model> [--n 5 --temperature 0.7]   # greedy + n samples via llama-server (CPU)
  python scripts/data/verify.py eval/vote/<model>.cands.jsonl eval/vote/<model>.cands.v.jsonl
  python eval/vote.py pick <model> [--fallback]                  # writes eval/results/<model>-vote[fb].jsonl

Answers vote by their sandbox output (status ok with non-empty stdout), otherwise by their normalized text.
Ties go to the group holding the greedy answer, which is also the answer when every group has one vote.
"""
import argparse
import collections
import json
import subprocess
import time

import pandas as pd
import requests

from generate import PORT, ROOT, SERVER, clean, engine, engine_user, wait_ready

VOTE = ROOT / "eval/vote"
BROKEN = ("syntax", "bad_usage", "missing_tool", "crashed_sandbox", "timeout")
FIXABLE = ("syntax", "bad_usage", "missing_tool")  # errors that show the command itself is wrong (engine.broken)


def sample(args):
    VOTE.mkdir(exist_ok=True)
    gguf = ROOT / f"models/gguf/{args.model}/{args.model}-Q8_0.gguf"
    test = pd.read_csv(ROOT / "data/raw/NL2SH-ALFA/test.csv")
    server = subprocess.Popen(["taskset", "-c", "0-7", str(SERVER), "-m", str(gguf), "--port", str(PORT), "-t", "8",
                               "-c", "4096", "--jinja"], stdout=subprocess.DEVNULL,
                              stderr=open(ROOT / "logs/llama-server.log", "w"))
    try:
        wait_ready()
        with open(VOTE / f"{args.model}.samples.jsonl", "w") as f, open(VOTE / f"{args.model}.cands.jsonl", "w") as c:
            for i, row in test.iterrows():
                messages = [{"role": "system", "content": engine.SYSTEM},
                            {"role": "user", "content": engine_user(row.nl, None, "none")}]
                t0 = time.perf_counter()
                cmds = []
                for k in range(args.n + 1):  # k == 0 is the greedy answer
                    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json={
                        "messages": messages, "max_tokens": 128, "seed": k,
                        "temperature": 0 if k == 0 else args.temperature}, timeout=120).json()
                    cmds.append(clean(r["choices"][0]["message"]["content"] or ""))
                f.write(json.dumps({"index": int(i), "nl": row.nl, "gold": row.bash, "gold2": row.bash2,
                                    "difficulty": int(row.difficulty), "cands": cmds,
                                    "latency_s": round(time.perf_counter() - t0, 3)}) + "\n")
                for j, cmd in enumerate(dict.fromkeys(cmds)):
                    c.write(json.dumps({"id": f"{i}|{j}", "cmd": cmd}) + "\n")
    finally:
        server.terminate()
        server.wait()


def pick(args):
    verdict = {}
    for r in map(json.loads, open(VOTE / f"{args.model}.cands.v.jsonl")):
        verdict[(int(r["id"].split("|")[0]), r["cmd"])] = r["verify"]
    changed = fell_back = 0
    with open(ROOT / f"eval/results/{args.model}-vote{'fb' if args.fallback else ''}.jsonl", "w") as f:
        for row in map(json.loads, open(VOTE / f"{args.model}.samples.jsonl")):
            cands = row.pop("cands")
            groups = collections.defaultdict(list)
            for k, cmd in enumerate(cands):
                v = verdict.get((row["index"], cmd), {})
                if v.get("status") in BROKEN and k:
                    continue  # a sample that cannot run gets no vote; the greedy answer always stays a candidate
                out = v.get("stdout", "").strip()
                key = ("out", v["status"], out) if v.get("status") == "ok" and out else ("cmd", " ".join(cmd.split()))
                groups[key].append((k, cmd))
            ranked = sorted(groups.values(), key=lambda g: (len(g), any(k == 0 for k, _ in g)), reverse=True)
            reps = [next((c for k, c in g if k == 0), collections.Counter(c for _, c in g).most_common(1)[0][0])
                    for g in ranked]
            cmd = reps[0]
            if args.fallback and verdict.get((row["index"], cmd), {}).get("status") in FIXABLE:
                # as engine.suggest: a best answer the sandbox showed broken loses to the next one that ran cleanly
                cmd = next((c for c in reps[1:] if verdict.get((row["index"], c), {}).get("status") not in FIXABLE),
                           cmd)
                fell_back += cmd != reps[0]
            changed += cmd != cands[0]
            f.write(json.dumps({**row, "raw": cmd, "cmd": cmd, "greedy": cands[0]}) + "\n")
    print("vote changed", changed, "of 300 answers" + (f"; fallback replaced {fell_back}" if args.fallback else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "pick"])
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--fallback", action="store_true", help="pick: replace a best answer that is broken in the sandbox "
                                                            "(writes <model>-votefb.jsonl)")
    a = ap.parse_args()
    sample(a) if a.mode == "sample" else pick(a)
