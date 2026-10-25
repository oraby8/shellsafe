# shellsafe

Describe what you want in plain English and get a Bash command back, with what each flag does, a safety rating and,
if you want, its real output from a sandbox. The model is a 0.5B fine-tune that runs on CPU; commands are never run on
your machine.

```
$ shellsafe ask "list all python files in this folder sorted by size"
╭─ best ─────────────────────────────────────────────────────────╮
│ find . -name '*.py' -exec du -h {} \; | sort -hr               │
│                                                                │
│ ● safe                                                         │
│                                                                │
│   sort -h  compare human readable numbers (e.g., 2K 1G)        │
│   sort -r  reverse the result of comparisons                   │
╰────────────────────────────────────────────────────────────────╯
```

- Model: [ahmedsamirtarjama/shellsafe-qwen2.5-coder-0.5b](https://huggingface.co/ahmedsamirtarjama/shellsafe-qwen2.5-coder-0.5b)
- Training data: [ahmedsamirtarjama/shellsafe-bash-sft](https://huggingface.co/datasets/ahmedsamirtarjama/shellsafe-bash-sft)

## Results

NL2SH-ALFA test set (300 tasks), InterCode-ALFA harness: each command runs in Docker and an LLM judge compares it
with the reference. All rows use the same harness and judge.

| Model | Accuracy |
|---|---|
| shellsafe (this repo: 0.5B + sandbox vote + fallback) | **53.7%** |
| shellsafe model alone, greedy | 50.0% |
| Qwen3-Coder-30B-A3B (teacher, for reference) | 57.7% |
| kitty-bash-llm (best off-the-shelf model under 1B we found) | 46.0% |
| Qwen2.5-Coder-0.5B-Instruct (base) | 25.3% |

The experiment logs in [eval/BASELINE.md](eval/BASELINE.md), [eval/CONTEXT.md](eval/CONTEXT.md) and
[eval/FINETUNE.md](eval/FINETUNE.md) cover what worked and what did not (context retrieval, DPO, GRPO, more data).

## Install

Needs Python 3.12, [uv](https://docs.astral.sh/uv/) and git. Docker is optional (sandbox vote and `--try`).

```bash
git clone <this repo> shellsafe && cd shellsafe
scripts/setup.sh                 # venv + tldr pages
.venv/bin/shellsafe ask "find files changed in the last day"
```

The first `ask` downloads the model from the Hugging Face Hub (about 0.5 GB, once). The model page asks you to accept
its terms once: open it, click agree (access is granted right away), then log in with `hf auth login` or set
`HF_TOKEN`. To use a GGUF file you already have, set `SHELLSAFE_MODEL` instead.

## Usage

| Command | What it does |
|---|---|
| `shellsafe ask "<request>"` | best command plus up to 2 alternatives, each with flags explained and a safety rating |
| `ask --try` | also run the best command in a read-only sandbox of the current folder and show its output |
| `ask --fast` | one answer, no sampling or sandbox vote: about 5x faster, a few points less accurate |
| `ask -n 1` | only the best command |
| `shellsafe check "<command>"` | rate how risky a command is, without running it |
| `shellsafe learn <tool> [--docs FILE]` | teach a tool that tldr does not cover, from its `--help`/man page or a docs file |
| `shellsafe daemon start` / `stop` / `status` | optional: keep the model loaded so `ask` takes about 1.4s instead of 5s |

The daemon never starts on its own. `ask` uses it only while it runs; `--no-daemon` skips it.

## How it works

1. The model writes one greedy answer and 5 sampled ones.
2. If Docker is available, every answer rated safe runs in a throwaway container with no network and the current
   folder mounted read-only. Answers that print the same output vote together; ties go to the greedy answer.
3. A winner that fails with a bad flag, a missing tool or a syntax error drops behind the next answer that ran
   cleanly, and stays visible with its error.
4. Each flag is explained from the man page or tldr; flags no doc mentions are flagged as possibly invented, and
   `safety.py` rates the command (safe, caution, dangerous) with its reasons.

The model only writes commands; ranking, explanations and the safety rating are plain code.

## Configuration

| Variable | Default |
|---|---|
| `SHELLSAFE_MODEL` | local `models/gguf/qwen05-v45/qwen05-v45-Q8_0.gguf`, else the Hub model |
| `SHELLSAFE_TLDR` | `third_party/tldr/pages` |
| `SHELLSAFE_SANDBOX_IMAGE` | `ubuntu:24.04` |
| `SHELLSAFE_HOME` | `~/.local/share/shellsafe` (learned tools, daemon socket and log) |
| `SHELLSAFE_CONTEXT` | `0`; `1` puts retrieved docs and examples in the prompt (helps third-party models, hurts ours) |

## Repo layout

```
src/shellsafe/   the CLI: engine (generate, vote, fallback), safety rules, sandbox, docs lookup, daemon
tests/           pytest; Docker and model tests skip when those are missing
sandbox/         verification image and fixture file system used to check training data
scripts/         data generation, training (SFT, DPO, GRPO), merging, GGUF conversion; run_*.sh are the pipelines
eval/            benchmark generation and scoring (NL2SH-ALFA + InterCode-ALFA), sandbox vote, error analysis
```

## Reproducing the model

`scripts/setup.sh --research` installs the training and eval dependencies, builds llama.cpp and downloads the raw
data and base models. The full pipeline needs a CUDA GPU (two for the data rounds), the local teacher
(`scripts/setup_teacher.sh`, `scripts/serve_teacher.sh`), and Docker with the verification image
(`docker build -t shellsafe-verify:1 sandbox/`). Scoring uses an Ollama judge (`qwen3.8`).

The final model came from `scripts/run_targeted_data.sh` and `run_targeted_train.sh` (v4), `run_targeted_r3.sh` (v5)
and `run_combine.sh` (weight average and vote). `scripts/data/relabel.py` and `scripts/grpo/reward.py` call
OpenAI-compatible API models and read `LLMX_BASE_URL` and `LLMX_API_KEY` from a `.llmx.env` file in the repo root;
that file is git-ignored.

Rules the results depend on: NL2SH-ALFA test items are never used for training or sent to API models, and
InterCode-Corrections is never used for training.

## Limitations

- About half of the hard multi-step requests still get a wrong command, often one that runs fine but does something
  slightly different. Read a command before you run it.
- GNU/Linux Bash; macOS/BSD flags can differ.
- English only.
