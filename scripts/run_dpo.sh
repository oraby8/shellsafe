#!/usr/bin/env bash
# Execution-checked DPO on top of v2: sample -> sandbox -> pairs -> DPO -> GGUF -> eval (deterministic judge).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=data/dpo; BASE=models/ft/qwen05-v2/final; NAME=qwen05-dpo
log() { echo "[$(date +%H:%M:%S)] $*"; }
mkdir -p $S

log "sample"
[ -s $S/samples.jsonl ] || CUDA_VISIBLE_DEVICES=1 .venv-vllm/bin/python scripts/dpo/sample.py \
  $BASE data/sft_v2_noctx/train.jsonl $S/samples.jsonl --prompts 40000 > logs/dpo_sample.log 2>&1
$PY scripts/dpo/pairs.py exec $S/samples.jsonl $S/exec_in.jsonl
log "execute $(wc -l < $S/exec_in.jsonl) commands"
$PY scripts/data/verify.py $S/exec_in.jsonl $S/exec_out.jsonl --workers 48 2>&1 | tail -1
$PY scripts/dpo/pairs.py build $S/samples.jsonl $S/exec_out.jsonl data/dpo_v1
log "train"
CUDA_VISIBLE_DEVICES=1 $PY scripts/dpo/train_dpo.py --data data/dpo_v1 --base $BASE --out models/ft/$NAME \
  > logs/train_dpo.log 2>&1
bash scripts/to_gguf.sh models/ft/$NAME/final $NAME >/dev/null 2>&1
log "eval"
$PY eval/generate.py $NAME --engine-prompt --context none
$PY eval/score.py eval/results/$NAME-Q8_0.jsonl --out eval/det/$NAME-Q8_0.scored.jsonl 2>&1 | tail -1
log "DPO_DONE"
