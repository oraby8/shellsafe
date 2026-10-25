#!/usr/bin/env bash
# DPO round 2: API-judge runnable-but-different samples, merge with execution pairs, DPO on v2, eval (cached judge).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=data/dpo; BASE=models/ft/qwen05-v2/final; NAME=qwen05-dpo-judged
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "judge"
$PY scripts/dpo/judge_pairs.py judge $S/samples.jsonl $S/exec_out.jsonl $S/judged.jsonl --limit 40000 --concurrency 48 2>&1 | tail -1
$PY scripts/dpo/judge_pairs.py build $S/samples.jsonl $S/judged.jsonl data/dpo_v1 data/dpo_v2
log "train"
CUDA_VISIBLE_DEVICES=1 $PY scripts/dpo/train_dpo.py --data data/dpo_v2 --base $BASE --out models/ft/$NAME \
  --lr 5e-6 --epochs 2 > logs/train_dpo_judged.log 2>&1
bash scripts/to_gguf.sh models/ft/$NAME/final $NAME >/dev/null 2>&1
log "eval"
$PY eval/generate.py $NAME --engine-prompt --context none
$PY eval/score.py eval/results/$NAME-Q8_0.jsonl --out eval/cached/$NAME.scored.jsonl 2>&1 | tail -1
log "DPOJ_DONE"
