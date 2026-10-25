#!/usr/bin/env bash
# Generate (CPU, 8 threads) then score (Docker + local judge) every baseline model.
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS=$(.venv/bin/python -c 'import json;print(" ".join(json.load(open("eval/models.json"))["models"]))')
for m in $MODELS; do
  .venv/bin/python eval/generate.py "$m" --quant "${QUANT:-Q8_0}" || echo "GEN FAIL $m"
done
for m in $MODELS; do
  .venv/bin/python eval/score.py "eval/results/$m-${QUANT:-Q8_0}.jsonl" 2>&1 | tail -1
done
