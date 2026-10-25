#!/usr/bin/env bash
# Option 3: GRPO on v2 with the sandbox + API-judge reward; eval the step-200 checkpoint and the final model.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; NAME=qwen05-grpo
CUDA_VISIBLE_DEVICES=1 $PY scripts/grpo/train_grpo.py --base models/ft/qwen05-v2/final --out models/ft/$NAME \
  --steps 400 --lr 5e-6 > logs/train_grpo.log 2>&1
for ck in checkpoint-200 final; do
  n=$NAME-${ck#checkpoint-}
  cp models/ft/qwen05-v2/final/tokenizer* models/ft/qwen05-v2/final/chat_template.jinja models/ft/$NAME/$ck/ 2>/dev/null || true
  bash scripts/to_gguf.sh models/ft/$NAME/$ck $n >/dev/null 2>&1
  $PY eval/generate.py $n --engine-prompt --context none
  $PY eval/score.py eval/results/$n-Q8_0.jsonl --out eval/cached/$n.scored.jsonl 2>&1 | tail -1
done
echo GRPO_DONE
