"""SFT set that upweights the training requests the current model gets wrong.

Usage: python scripts/hard/build_hard_sft.py SFT_DIR JUDGED.jsonl EXEC_PAIRS_DIR OUT_DIR [--repeat 3]
Hard = a sampled answer was judged wrong by two API models, or broke/errored in the sandbox (data/dpo_v1).
OUT train = the full SFT train set plus each hard example repeated --repeat extra times; dev is copied as is.
"""
import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sft_dir")
    ap.add_argument("judged")
    ap.add_argument("exec_pairs")
    ap.add_argument("out_dir")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    hard = set()
    for r in map(json.loads, open(args.judged)):
        if list(r["votes"].values()).count("incorrect") >= 2:
            hard.add(r["id"].rsplit("|", 1)[0])
    for split in ("train", "dev"):
        hard |= {json.loads(l)["id"] for l in open(Path(args.exec_pairs) / f"{split}.jsonl")}
    train = [json.loads(l) for l in open(Path(args.sft_dir) / "train.jsonl")]
    extra = [r for r in train if r["id"] in hard] * args.repeat
    rows = train + extra
    random.Random(0).shuffle(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    shutil.copy(Path(args.sft_dir) / "dev.jsonl", out / "dev.jsonl")
    print(f"hard requests {len(hard)}, in train {len(extra) // args.repeat}, train rows {len(rows)}")


if __name__ == "__main__":
    main()
