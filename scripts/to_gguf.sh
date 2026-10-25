#!/usr/bin/env bash
# Convert a Hugging Face model directory to GGUF (f16, Q8_0, Q4_K_M) for CPU inference.
# Usage: scripts/to_gguf.sh models/base/<name> [output name]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LLAMA=$ROOT/third_party/llama.cpp
src=$1
name=${2:-$(basename "$src")}
out=$ROOT/models/gguf/$name
mkdir -p "$out"
[ -f "$out/$name-f16.gguf" ] || PYTHONPATH=$LLAMA/gguf-py $ROOT/.venv/bin/python "$LLAMA/convert_hf_to_gguf.py" "$src" --outtype f16 --outfile "$out/$name-f16.gguf"
for q in Q8_0 Q4_K_M; do
  [ -f "$out/$name-$q.gguf" ] || "$LLAMA/build-cpu/bin/llama-quantize" "$out/$name-f16.gguf" "$out/$name-$q.gguf" "$q" >/dev/null
done
ls -la "$out"
