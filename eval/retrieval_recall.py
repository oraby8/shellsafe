"""How often does the lookup return the page for the command(s) the reference answer uses?"""
import sys

import pandas as pd

from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(_ROOT / "src"))
from shellsafe.retrieve import ExampleIndex, Retriever, TldrIndex, command_names  # noqa: E402

DATA = str(_ROOT / "data/raw/NL2SH-ALFA")
test = pd.read_csv(f"{DATA}/test.csv")
train = pd.read_csv(f"{DATA}/train.csv").dropna()

# Leakage check: identical request text in train and test would inflate every later number.
overlap = set(test.nl.str.lower().str.strip()) & set(train.nl.str.lower().str.strip())
print("test requests also in train (exact):", len(overlap))
train = train[~train.nl.str.lower().str.strip().isin(overlap)]

tldr = TldrIndex()
retr = Retriever(tldr, ExampleIndex(list(zip(train.nl, train.bash))))
methods = {"tldr-only": lambda q, k: tldr.search(q, k), "examples+tldr": retr.commands}
for label, fn in methods.items():
    for k in (1, 3, 5):
        main_hit = all_hit = n = 0
        for row in test.itertuples():
            gold = command_names(row.bash)
            if not gold:
                continue
            n += 1
            hits = {p.name for p in fn(row.nl, k)}
            main_hit += gold[0] in hits
            all_hit += all(g in hits for g in gold)
        print(f"{label:14s} k={k}: main command {main_hit/n:.1%}, every command {all_hit/n:.1%} (n={n})")
