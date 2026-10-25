#!/usr/bin/env bash
# Option 2: continue SFT from v2 with the requests it gets wrong upweighted, then eval with the cached judge.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; NAME=qwen05-hard
$PY scripts/hard/build_hard_sft.py data/sft_v2_noctx data/dpo/judged.jsonl data/dpo_v1 data/sft_v3_hard
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_sft.py --data data/sft_v3_hard --base models/ft/qwen05-v2/final \
  --out models/ft/$NAME --epochs 1 --lr 1e-5 > logs/train_hard.log 2>&1
bash scripts/to_gguf.sh models/ft/$NAME/final $NAME >/dev/null 2>&1
$PY eval/generate.py $NAME --engine-prompt --context none
$PY eval/score.py eval/results/$NAME-Q8_0.jsonl --out eval/cached/$NAME.scored.jsonl 2>&1 | tail -1
echo HARD_DONE
