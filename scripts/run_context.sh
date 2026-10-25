#!/usr/bin/env bash
# Context experiment: does showing tldr docs / similar examples help sub-1B models? Judge without thinking.
set -uo pipefail
cd "$(dirname "$0")/.."
RUNS=(
  "kitty-bash-llm docs" "kitty-bash-llm examples" "kitty-bash-llm both"
  "Qwen2.5-Coder-0.5B-Instruct none" "Qwen2.5-Coder-0.5B-Instruct docs" "Qwen2.5-Coder-0.5B-Instruct examples" "Qwen2.5-Coder-0.5B-Instruct both"
  "shellminator-qwen05b-dpo-selfplay none" "shellminator-qwen05b-dpo-selfplay both"
)
mkdir -p eval/nothink
for run in "${RUNS[@]}"; do
  set -- $run
  [ "$2" = none ] || .venv/bin/python eval/generate.py "$1" --context "$2" || echo "GEN FAIL $run"
done
cp eval/judge_check/kitty-nothink.scored.jsonl eval/nothink/kitty-bash-llm-Q8_0.scored.jsonl
for run in "${RUNS[@]}"; do
  set -- $run
  f="$1-Q8_0"; [ "$2" = none ] || f="$f-ctx_$2"
  .venv/bin/python eval/score.py "eval/results/$f.jsonl" --out "eval/nothink/$f.scored.jsonl" 2>&1 | tail -1
done
echo CONTEXT_DONE
