"""Full fine-tune of a small chat model on data/sft-style jsonl (loss on the assistant command only).

Usage: python scripts/train_sft.py --data data/sft --out models/ft/NAME [--base models/base/Qwen2.5-Coder-0.5B-Instruct]
Non-tldr rows (NL2SH, compose, fixes) are repeated --upweight times: they match how real users phrase requests,
while tldr-derived rows dominate the raw counts.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainingArguments)


def encode(tok, ex, max_len):
    prompt = tok.apply_chat_template(ex["messages"][:-1], tokenize=False, add_generation_prompt=True)
    answer = ex["messages"][-1]["content"] + tok.eos_token + "\n"
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(answer, add_special_tokens=False)["input_ids"]
    p_ids = p_ids[-(max_len - len(a_ids)):]  # keep the request end of an over-long context
    return {"input_ids": p_ids + a_ids, "attention_mask": [1] * (len(p_ids) + len(a_ids)),
            "labels": [-100] * len(p_ids) + a_ids}


def load(path, tok, max_len, upweight=1):
    rows = [json.loads(l) for l in open(path)]
    if upweight > 1:
        rows += [r for r in rows if not r["source"].startswith("tldr")] * (upweight - 1)
    random.Random(0).shuffle(rows)
    return [encode(tok, r, max_len) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="models/base/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--upweight", type=int, default=2)
    ap.add_argument("--grad-ckpt", action="store_true", help="trade speed for memory")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    train = load(Path(args.data) / "train.jsonl", tok, args.max_len, args.upweight)
    dev = load(Path(args.data) / "dev.jsonl", tok, args.max_len)
    print(f"train {len(train)}  dev {len(dev)}  "
          f"max tokens {max(len(x['input_ids']) for x in train)}")
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=0.03, weight_decay=0.0, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.accum, bf16=True,
        train_sampling_strategy="group_by_length", use_liger_kernel=True, logging_steps=20, eval_strategy="steps", eval_steps=300, save_strategy="steps",
        save_steps=300, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="none", dataloader_num_workers=2, gradient_checkpointing=args.grad_ckpt)
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=dev,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    trainer.train()
    trainer.save_model(args.out + "/final")
    tok.save_pretrained(args.out + "/final")
    print("eval", trainer.evaluate())


if __name__ == "__main__":
    main()
