"""Ceiling check only: answer the NL2SH-ALFA test prompts with a gateway model (eval/generate.py's prompt and clean()).

Usage: python eval/gateway_generate.py moonshotai/Kimi-K3 [--concurrency 16]
Writes eval/results/<short-name>-gateway.jsonl for eval/score.py. These answers are for scoring only and must never
feed a training set.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from generate import ROOT, clean  # noqa: E402

ENV = dict(l.strip().split("=", 1) for l in open(ROOT / ".llmx.env") if "=" in l)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    client = OpenAI(base_url=ENV["LLMX_BASE_URL"], api_key=ENV["LLMX_API_KEY"], timeout=600, max_retries=5)
    prompt = json.loads((ROOT / "eval/models.json").read_text())["default"]
    test = pd.read_csv(ROOT / "data/raw/NL2SH-ALFA/test.csv")
    out = ROOT / f"eval/results/{args.model.split('/')[-1]}-gateway.jsonl"

    def ask(item):
        i, row = item
        messages = [{"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user_tmpl"].format(nl=row.nl)}]
        t0 = time.perf_counter()
        r = client.chat.completions.create(model=args.model, messages=messages, temperature=0,
                                           max_completion_tokens=4000)
        raw = r.choices[0].message.content or ""
        return {"index": int(i), "nl": row.nl, "gold": row.bash, "gold2": row.bash2,
                "difficulty": int(row.difficulty), "raw": raw, "cmd": clean(raw),
                "latency_s": round(time.perf_counter() - t0, 3),
                "prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens}

    with ThreadPoolExecutor(args.concurrency) as pool:
        rows = list(pool.map(ask, test.iterrows()))
    with out.open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("wrote", out, "empty", sum(not r["cmd"] for r in rows))


if __name__ == "__main__":
    main()
