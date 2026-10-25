#!/usr/bin/env bash
# v2: wait for the API relabel, apply it (sandbox-check replacements), retrain the no-context model, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; D=scripts/data; S=data/stage
log() { echo "[$(date +%H:%M:%S)] $*"; }

while pgrep -f "[r]elabel.py data/stage/final" >/dev/null; do sleep 60; done
log "relabel rows: $(wc -l < $S/relabeled.jsonl) / $(wc -l < $S/final.jsonl)"
$PY $D/apply_relabel.py candidates $S/relabeled.jsonl $S/relabel_repl.jsonl
log "verify $(wc -l < $S/relabel_repl.jsonl) replacements"
$PY $D/verify.py $S/relabel_repl.jsonl $S/relabel_repl.verified.jsonl --workers 24 2>&1 | tail -1
$PY $D/apply_relabel.py merge $S/relabeled.jsonl $S/relabel_repl.verified.jsonl $S/final_v2.jsonl
log "format"
$PY $D/format_sft.py $S/final_v2.jsonl data/sft_v2_noctx none
log "train"
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_sft.py --data data/sft_v2_noctx --out models/ft/qwen05-v2 > logs/train_v2.log 2>&1
bash scripts/to_gguf.sh models/ft/qwen05-v2/final qwen05-v2 >/dev/null 2>&1
log "eval"
$PY eval/generate.py qwen05-v2 --engine-prompt --context none
$PY eval/score.py eval/results/qwen05-v2-Q8_0.jsonl --out eval/nothink/qwen05-v2-Q8_0.scored.jsonl 2>&1 | tail -1
log "V2_DONE"
