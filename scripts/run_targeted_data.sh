#!/usr/bin/env bash
# Targeted data (scripts/targeted/gen.py) with two local teacher copies, one per GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; T=data/targeted; mkdir -p $T
log() { echo "[$(date +%H:%M:%S)] $*"; }
TEACHER_GPU=0 TEACHER_PORT=8100 bash scripts/serve_teacher.sh > logs/teacher0.log 2>&1 &
TEACHER_GPU=1 TEACHER_PORT=8101 bash scripts/serve_teacher.sh > logs/teacher1.log 2>&1 &
trap 'pkill -f "vllm serve models/teacher" || true' EXIT
for p in 8100 8101; do until curl -sf localhost:$p/v1/models >/dev/null; do sleep 10; done; done
log "make"
$PY scripts/targeted/gen.py make $T/gen2.jsonl $T/gen.jsonl --calls 6000 2>&1 | grep -v "it/s" | tail -3
rm -f $T/ans.jsonl
log "answer $(wc -l < $T/gen2.ids.jsonl)"
$PY scripts/targeted/gen.py answer $T/gen2.ids.jsonl $T/ans.jsonl 2>&1 | grep -v "it/s" | tail -3
pkill -f "vllm serve models/teacher" || true
log "verify"
$PY scripts/data/verify.py $T/gen2.ids.jsonl $T/gen.v.jsonl --workers 48 2>&1 | tail -1
$PY scripts/data/verify.py $T/ans.jsonl $T/ans.v.jsonl --workers 48 2>&1 | tail -1
$PY scripts/targeted/gen.py build $T/gen.v.jsonl $T/ans.v.jsonl $T
log "TGT_DATA_DONE $(wc -l < $T/targeted.jsonl)"
