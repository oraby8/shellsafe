#!/usr/bin/env bash
# One-time setup. Default: the CLI (venv + tldr pages for flag explanations).
# --research also installs the training/eval extras, builds llama.cpp (GGUF conversion and llama-server) and
# downloads the raw datasets and base models (scripts/download.py).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "needs uv: https://docs.astral.sh/uv/"; exit 1; }
mkdir -p logs third_party

# Pinned to the commits the published results were made with.
clone() {  # dir repo commit
  [ -d "third_party/$1" ] && return
  git clone -q --filter=blob:none "https://github.com/$2" "third_party/$1"
  git -C "third_party/$1" checkout -q "$3"
}

[ -d .venv ] || uv venv -q --python 3.12 .venv
if [ "${1:-}" = "--research" ]; then
  uv pip install -q --python .venv/bin/python -e ".[train,eval,dev]"
else
  uv pip install -q --python .venv/bin/python -e ".[dev]"
fi
clone tldr tldr-pages/tldr 5d3655b552f8fee7b9ddb4b6534e327c0933c326

if [ "${1:-}" = "--research" ]; then
  clone llama.cpp ggml-org/llama.cpp 7775ffb7e48fa0c7dcffd5815743b3d2fb3b6761
  [ -x third_party/llama.cpp/build-cpu/bin/llama-server ] || {
    cmake -S third_party/llama.cpp -B third_party/llama.cpp/build-cpu -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build third_party/llama.cpp/build-cpu -j --target llama-server llama-quantize >/dev/null
  }
  .venv/bin/python scripts/download.py
fi
echo "ready: .venv/bin/shellsafe ask \"list the 5 biggest files here\""
