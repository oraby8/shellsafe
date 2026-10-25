"""Average the weights of fine-tunes that share a base model (a "model soup").

Usage: python scripts/merge.py OUT_DIR MODEL_DIR [MODEL_DIR ...]
Tokenizer and config files are copied from the first model.
"""
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

out, srcs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
out.mkdir(parents=True, exist_ok=True)
states = [load_file(p / "model.safetensors") for p in srcs]
merged = {k: (sum(s[k].float() for s in states) / len(states)).to(states[0][k].dtype) for k in states[0]}
save_file(merged, out / "model.safetensors", metadata={"format": "pt"})
for f in srcs[0].iterdir():
    if f.name not in ("model.safetensors", "training_args.bin"):
        shutil.copy(f, out / f.name)
print("merged", len(srcs), "models into", out)
