#!/usr/bin/env bash
# After both fine-tunes finish: GGUF-convert them and score on NL2SH-ALFA test (CPU, no-think judge).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
until [ -f models/ft/qwen05-ctx/final/config.json ] && [ -f models/ft/qwen05-noctx/final/config.json ]; do sleep 60; done
for m in qwen05-ctx qwen05-noctx; do bash scripts/to_gguf.sh models/ft/$m/final $m >/dev/null; done
RUNS=("qwen05-noctx none" "qwen05-ctx both" "qwen05-ctx none")
for run in "${RUNS[@]}"; do
  set -- $run
  $PY eval/generate.py "$1" --engine-prompt --context "$2" || echo "GEN FAIL $run"
done
for run in "${RUNS[@]}"; do
  set -- $run
  f="$1-Q8_0"; [ "$2" = none ] || f="$f-ctx_$2"
  echo "== $f"; $PY eval/score.py "eval/results/$f.jsonl" --out "eval/nothink/$f.scored.jsonl" 2>&1 | tail -3
done
echo FT_EVAL_DONE
