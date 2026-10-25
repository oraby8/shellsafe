#!/usr/bin/env bash
# Two uses of data/targeted/targeted.jsonl, one per GPU:
#   qwen05-v4     v2's recipe from the base model on v2 data + targeted rows
#   qwen05-v2tgt  continue v2 on targeted rows + a 20k sample of v2 data (against forgetting)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; T=data/targeted/targeted.jsonl
mkdir -p data/sft_v4 data/sft_v2tgt
cat data/sft_v2_noctx/train.jsonl $T > data/sft_v4/train.jsonl
cp data/sft_v2_noctx/dev.jsonl data/sft_v4/dev.jsonl
{ cat $T; shuf -n 20000 --random-source=<(yes) data/sft_v2_noctx/train.jsonl; } > data/sft_v2tgt/train.jsonl
cp data/sft_v2_noctx/dev.jsonl data/sft_v2tgt/dev.jsonl
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_sft.py --data data/sft_v4 --out models/ft/qwen05-v4 > logs/train_v4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY scripts/train_sft.py --data data/sft_v2tgt --base models/ft/qwen05-v2/final \
  --out models/ft/qwen05-v2tgt --epochs 1 --lr 1e-5 > logs/train_v2tgt.log 2>&1 &
wait
for NAME in qwen05-v2tgt qwen05-v4; do
  bash scripts/to_gguf.sh models/ft/$NAME/final $NAME >/dev/null 2>&1
  $PY eval/generate.py $NAME --engine-prompt --context none
  $PY eval/score.py eval/results/$NAME-Q8_0.jsonl --out eval/cached/$NAME.scored.jsonl 2>&1 | tail -1
done
echo TGT_TRAIN_DONE
