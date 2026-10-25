#!/usr/bin/env bash
# (1) weight-average v4 and v5; (2) majority vote over sandbox outputs for v4 and the merge.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY scripts/merge.py models/ft/qwen05-v45/final models/ft/qwen05-v4/final models/ft/qwen05-v5/final
bash scripts/to_gguf.sh models/ft/qwen05-v45/final qwen05-v45 >/dev/null 2>&1
$PY eval/generate.py qwen05-v45 --engine-prompt --context none
$PY eval/score.py eval/results/qwen05-v45-Q8_0.jsonl --out eval/cached/qwen05-v45.scored.jsonl 2>&1 | tail -1
for NAME in qwen05-v4 qwen05-v45; do
  (cd eval && ../$PY vote.py sample $NAME)
  $PY scripts/data/verify.py eval/vote/$NAME.cands.jsonl eval/vote/$NAME.cands.v.jsonl --workers 32 2>&1 | tail -1
  (cd eval && ../$PY vote.py pick $NAME)
  $PY eval/score.py eval/results/$NAME-vote.jsonl --out eval/cached/$NAME-vote.scored.jsonl 2>&1 | tail -1
done
echo COMBINE_DONE
