# Baseline — 2026-10-22

NL2SH-ALFA test set (300 prompts), InterCode-ALFA execution scoring, judge `qwen3.8:latest` via Ollama.
Inference: llama.cpp CPU build, GGUF Q8_0, 8 threads pinned (`taskset -c 0-7`), greedy, one pass.
Easy/Medium/Hard = dataset difficulty 0/1/2.

| Model | Params | Fine-tuned for bash | Acc | Easy | Medium | Hard | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| kitty-bash-llm | 0.5B | yes | **48.3%** | 66% | 49% | 30% | 0.16s | 0.33s |
| shellminator-qwen05b-dpo-selfplay | 0.5B | yes (SFT+DPO) | 47.0% | 68% | 47% | 26% | 0.15s | 0.38s |
| nl-to-bash (NL2SH-ALFA train) | 0.5B | yes | 40.3% | 53% | 44% | 24% | 0.15s | 0.30s |
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | no | 25.7% | 47% | 17% | 13% | 0.16s | 0.41s |
| Qwen3.5-0.8B | 0.8B | no | 23.0% | 29% | 21% | 19% | 0.40s | 2.14s |
| Qwen2.5-Coder-0.5B-Instruct-NL2SH (paper authors) | 0.5B | yes | 22.3% | 33% | 22% | 12% | 0.25s | 1.19s |
| Qwen3-0.6B | 0.6B | no | 11.3% | 17% | 11% | 6% | 0.18s | 0.40s |

Reproduce: `scripts/run_baseline.sh` (per-row outputs in `eval/results/*.scored.jsonl`, failures via `eval/failures.py`).
