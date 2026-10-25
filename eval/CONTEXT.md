# Context experiment — 2026-10-22

Question: does putting retrieved tldr docs and/or similar solved examples in the prompt help sub-1B models
that were not trained with context? Same setup as BASELINE.md but judge without thinking (stricter by ~3 points;
compare only within this table). Retrieval: examples+tldr voting, top-3 pages, top-5 similar examples.

| Model | none | docs | examples | both |
|---|---|---|---|---|
| kitty-bash-llm | **45.0%** | 38.7% | 35.0% | 32.7% |
| shellminator-qwen05b-dpo-selfplay | **43.0%** | – | – | 38.3% |
| Qwen2.5-Coder-0.5B-Instruct | **25.7%** | 22.0% | 20.7% | 24.0% |

Latency p50 rises from ~0.16s (none) to 0.36–0.90s with context (prompt ~400–900 tokens on 8 cores).

Conclusion: untrained context hurts every model. Context must be part of fine-tuning (format_sft.py trains with
mixed context modes); compare a with-context fine-tune against a no-context fine-tune before shipping either.
