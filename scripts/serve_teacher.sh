#!/usr/bin/env bash
# OpenAI-compatible teacher on :8100, on GPU 0 only. The GPUs sit on different sockets with no P2P link,
# so tensor parallel across them hangs; GPU 1 stays free for the Ollama judge and for training.
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${TEACHER_GPU:-0}
exec .venv-vllm/bin/vllm serve models/teacher/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name teacher --port ${TEACHER_PORT:-8100} --host 127.0.0.1 \
  --gpu-memory-utilization 0.88 --max-model-len 8192 \
  --max-num-seqs 128 --enable-prefix-caching
