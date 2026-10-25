#!/usr/bin/env bash
# Re-score the main runs with the deterministic judge (temperature 0 actually applied). Writes eval/det/.
cd "$(dirname "$0")/.."
mkdir -p eval/det
for f in qwen05-noctx-Q8_0 qwen05-v2-Q8_0 kitty-bash-llm-Q8_0 shellminator-qwen05b-dpo-selfplay-Q8_0 Qwen2.5-Coder-0.5B-Instruct-Q8_0 qwen05-ctx-Q8_0-ctx_both; do
  .venv/bin/python eval/score.py eval/results/$f.jsonl --out eval/det/$f.scored.jsonl 2>&1 | tail -1
done
echo RESCORE_DONE
