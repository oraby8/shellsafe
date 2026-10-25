#!/usr/bin/env bash
# Separate venv for vLLM (it pins its own torch) + the teacher model weights.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH HF_HOME=$PWD/.hf_cache HF_HUB_ENABLE_HF_TRANSFER=1
[ -d .venv-vllm ] || uv venv -q --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python -q vllm hf_transfer 2>&1 | tail -3
.venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8', local_dir='models/teacher/Qwen3-Coder-30B-A3B-Instruct-FP8')
print('teacher downloaded')"
du -sh models/teacher/*
echo SETUP_TEACHER_DONE
