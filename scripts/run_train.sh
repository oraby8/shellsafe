#!/usr/bin/env bash
# Two fine-tunes side by side: with retrieved context (GPU 0) and without (GPU 1).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
[ -f data/sft_noctx/train.jsonl ] || $PY scripts/data/format_sft.py data/stage/final.jsonl data/sft_noctx none
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_sft.py --data data/sft --out models/ft/qwen05-ctx > logs/train_ctx.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_sft.py --data data/sft_noctx --out models/ft/qwen05-noctx > logs/train_noctx.log 2>&1 &
wait
echo TRAIN_DONE
