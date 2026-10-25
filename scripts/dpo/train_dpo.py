"""DPO on execution-checked preference pairs, starting from an SFT checkpoint (which is also the reference).

Usage: python scripts/dpo/train_dpo.py --data data/dpo_v1 --base models/ft/qwen05-v2/final --out models/ft/qwen05-dpo
An SFT term on the chosen command (loss_type sigmoid+sft) keeps the gold command's likelihood from drifting down,
a known DPO failure mode when chosen and rejected share most tokens.
"""
import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    data = load_dataset("json", data_files={"train": f"{args.data}/train.jsonl", "dev": f"{args.data}/dev.jsonl"})
    data = data.remove_columns(["id", "source"])
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    cfg = DPOConfig(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.05, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.accum, bf16=True, beta=args.beta, loss_type=["sigmoid", "sft"],
        loss_weights=[1.0, args.sft_weight], max_length=1024, logging_steps=20, eval_strategy="steps",
        eval_steps=200, save_strategy="steps", save_steps=200, save_total_limit=2, report_to="none")
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=data["train"], eval_dataset=data["dev"],
                         processing_class=tok)
    trainer.train()
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print("eval", trainer.evaluate())


if __name__ == "__main__":
    main()
