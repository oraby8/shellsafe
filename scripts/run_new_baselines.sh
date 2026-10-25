#!/usr/bin/env bash
# Baselines requested 2026-10-24. Not testable as models: incoai/Qwen3.8-Flash-Next-DFlash2 (speculative-decoding
# draft, not standalone) and zjj1993/Qwen3.8-Flash-Next-uncensored-china (a steering vector for a >1B model).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for repo in motexture/SmolLCoder-360M-Instruct google/functiongemma-270m-it MinimaLabs/KLM-1.5; do
  $PY -c "from huggingface_hub import snapshot_download as s; s('$repo', local_dir='models/base/${repo#*/}')" >/dev/null && echo "got $repo"
done
export CUDA_VISIBLE_DEVICES=0
run() { $PY eval/hf_generate.py models/base/$1 --mode $2 2>&1 | grep -E "wrote|Error|error" | tail -2
        $PY eval/score.py eval/results/$1-hf-$2.jsonl --out eval/cached/$1-hf-$2.scored.jsonl 2>&1 | tail -1; }
run SmolLCoder-360M-Instruct chat
run functiongemma-270m-it chat
run functiongemma-270m-it tools
run KLM-1.5 fewshot
run Qwen2.5-Coder-0.5B-Instruct fewshot
echo NEWBASE_DONE
