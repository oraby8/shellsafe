"""Sample N commands per training request from an SFT model with vLLM (run with .venv-vllm).

Usage: python scripts/dpo/sample.py MODEL_DIR SFT_TRAIN.jsonl OUT.jsonl [--prompts 30000] [--n 8]
All non-tldr requests are kept (they match how users phrase things); tldr rows fill the rest at random.
OUT rows: {"id", "messages" (system+user), "gold", "source", "samples": [unique commands != gold]}.
"""
import argparse
import json
import random

from vllm import LLM, SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("train")
    ap.add_argument("out")
    ap.add_argument("--prompts", type=int, default=30000)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.train)]
    rng = random.Random(0)
    other = [r for r in rows if not r["source"].startswith("tldr")]
    tldr = [r for r in rows if r["source"].startswith("tldr")]
    rng.shuffle(tldr)
    rows = other + tldr[:max(0, args.prompts - len(other))]

    llm = LLM(args.model, gpu_memory_utilization=0.3, max_model_len=2048, seed=0)
    params = SamplingParams(n=args.n, temperature=args.temperature, top_p=0.95, max_tokens=128, seed=0)
    outs = llm.chat([r["messages"][:-1] for r in rows], params, use_tqdm=True)
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            gold = r["messages"][-1]["content"]
            seen, samples = {" ".join(gold.split())}, []
            for c in o.outputs:
                cmd = c.text.strip()
                key = " ".join(cmd.split())
                if cmd and key not in seen and c.finish_reason == "stop":
                    seen.add(key)
                    samples.append(cmd)
            f.write(json.dumps({"id": r["id"], "messages": r["messages"][:-1], "gold": gold,
                                "source": r["source"], "samples": samples}) + "\n")


if __name__ == "__main__":
    main()
