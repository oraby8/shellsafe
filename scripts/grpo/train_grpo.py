"""GRPO from an SFT checkpoint with the sandbox + API-judge reward (scripts/grpo/reward.py).

Usage: python scripts/grpo/train_grpo.py --base models/ft/qwen05-v2/final --out models/ft/qwen05-grpo [--steps 400]
Prompts are training requests from data/dpo/samples.jsonl, mostly ones the SFT model got wrong at least once
(groups where every answer scores the same teach GRPO nothing), each with its gold command's sandbox verdict.
The KL term (beta) keeps the policy near the SFT model, against the drift seen with strong DPO.
"""
import argparse
import json
import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from reward import Reward

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[2]  # repo root
S = str(_ROOT / "data")


def prompts(n_easy):
    hard = {json.loads(l)["id"].rsplit("|", 1)[0] for l in open(f"{S}/dpo/judged.jsonl")
            if list(json.loads(l)["votes"].values()).count("incorrect") >= 2}
    for split in ("train", "dev"):
        hard |= {json.loads(l)["id"] for l in open(f"{S}/dpo_v1/{split}.jsonl")}
    gold_v = {}
    for l in open(f"{S}/dpo/exec_out.jsonl"):
        if "|g\"" in l:
            r = json.loads(l)
            gold_v[r["id"][:-2]] = r["verify"]
    rows, easy = [], []
    for r in map(json.loads, open(f"{S}/dpo/samples.jsonl")):
        g = gold_v.get(r["id"])
        if not g or g["status"] in ("syntax", "bad_usage", "crashed_sandbox", "worker_error"):
            continue
        row = {"prompt": r["messages"], "id": r["id"], "request": r["messages"][-1]["content"],
               "gold": r["gold"], "gold_v": json.dumps(g)}
        (rows if r["id"] in hard else easy).append(row)
    rng = random.Random(0)
    rng.shuffle(easy)
    rows += easy[:n_easy]
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--easy", type=int, default=3000)
    args = ap.parse_args()

    data = Dataset.from_list(prompts(args.easy))
    print("prompts", len(data))
    reward = Reward()

    def exec_judge_reward(prompts, completions, id, request, gold, gold_v, **_):
        rows = [{"id": i, "request": q, "gold": g, "gold_v": json.loads(gv), "cmd": c[0]["content"].strip()}
                for i, q, g, gv, c in zip(id, request, gold, gold_v, completions)]
        return reward.score(rows)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    per_device = args.prompts_per_step * args.gens // 2
    cfg = GRPOConfig(
        output_dir=args.out, max_steps=args.steps, learning_rate=args.lr, lr_scheduler_type="constant_with_warmup",
        warmup_steps=10, per_device_train_batch_size=per_device, gradient_accumulation_steps=2,
        num_generations=args.gens, max_completion_length=96, temperature=1.0, top_p=0.95, beta=args.beta,
        mask_truncated_completions=True, bf16=True, logging_steps=5, save_strategy="steps", save_steps=100,
        save_total_limit=4, report_to="none")
    trainer = GRPOTrainer(model=model, reward_funcs=exec_judge_reward, args=cfg, train_dataset=data,
                          processing_class=tok)
    trainer.train()
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")


if __name__ == "__main__":
    main()
