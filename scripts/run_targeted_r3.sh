#!/usr/bin/env bash
# Round 3: more hard find/pipeline data, half of the home-folder requests with bare names; then v5 = v4's recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; T=data/targeted
log() { echo "[$(date +%H:%M:%S)] $*"; }
TEACHER_GPU=0 TEACHER_PORT=8100 bash scripts/serve_teacher.sh > logs/teacher0.log 2>&1 &
TEACHER_GPU=1 TEACHER_PORT=8101 bash scripts/serve_teacher.sh > logs/teacher1.log 2>&1 &
trap 'pkill -f "vllm serve models/teacher" || true' EXIT
for p in 8100 8101; do until curl -sf localhost:$p/v1/models >/dev/null; do sleep 10; done; done
log "make"
$PY scripts/targeted/gen.py make $T/gen3.jsonl $T/gen.jsonl $T/gen2.jsonl --calls 6000 --prefix tgt3 --hard-frac 0.8 \
  --cats find_complex=35,multistep=35,file_content=10,abs_dirs=10,cwd_home=10 2>&1 | grep -v "it/s" | tail -3
log "answer $(wc -l < $T/gen3.ids.jsonl)"
$PY scripts/targeted/gen.py answer $T/gen3.ids.jsonl $T/ans3.jsonl 2>&1 | grep -v "it/s" | tail -3
pkill -f "vllm serve models/teacher" || true
sleep 10
log "verify"
$PY scripts/data/verify.py $T/gen3.ids.jsonl $T/gen3.v.jsonl --workers 48 2>&1 | tail -1
$PY scripts/data/verify.py $T/ans3.jsonl $T/ans3.v.jsonl --workers 48 2>&1 | tail -1
cat $T/gen.v.jsonl $T/gen3.v.jsonl > $T/all_gen.v.jsonl; cat $T/ans.v.jsonl $T/ans3.v.jsonl > $T/all_ans.v.jsonl
mkdir -p $T/r3
$PY scripts/targeted/gen.py build $T/all_gen.v.jsonl $T/all_ans.v.jsonl $T/r3
log "data $(wc -l < $T/r3/targeted.jsonl)"
mkdir -p data/sft_v5
cat data/sft_v2_noctx/train.jsonl $T/r3/targeted.jsonl > data/sft_v5/train.jsonl
cp data/sft_v2_noctx/dev.jsonl data/sft_v5/dev.jsonl
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_sft.py --data data/sft_v5 --out models/ft/qwen05-v5 > logs/train_v5.log 2>&1
bash scripts/to_gguf.sh models/ft/qwen05-v5/final qwen05-v5 >/dev/null 2>&1
$PY eval/generate.py qwen05-v5 --engine-prompt --context none
$PY eval/score.py eval/results/qwen05-v5-Q8_0.jsonl --out eval/cached/qwen05-v5.scored.jsonl 2>&1 | tail -1
log "R3_DONE"
