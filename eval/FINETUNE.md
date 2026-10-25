# Fine-tune v1 (Qwen2.5-Coder-0.5B-Instruct, full SFT, 2 epochs, 62k pairs)

Scored on NL2SH-ALFA test (300) with the no-think Ollama judge; Q8_0 GGUF on 8 CPU threads.

| Model | Acc | d0 / d1 / d2 | p50 |
|---|---|---|---|
| Qwen2.5-Coder-0.5B-Instruct (base) | 25.7% | | 0.15s |
| kitty-bash-llm (best off-the-shelf) | 45.0% | | 0.15s |
| **qwen05-noctx** | **46.3%** | 0.64 / 0.41 / 0.34 | 0.17s |
| qwen05-ctx, docs + examples | 38.7% | 0.46 / 0.37 / 0.33 | 1.02s |
| qwen05-ctx, no context | 40.0% | 0.55 / 0.35 / 0.30 | 0.15s |

Context still hurts after training on it. Likely cause: tldr-derived training requests were generated from the very
page that retrieval returns, so the model learned to copy from docs; at test time retrieval finds the right page only
~70% of the time. Decision: the model gets no context; the CLI keeps docs for flag explanations and validation.

Dev loss: noctx 0.880, ctx ~0.92.

## 2026-10-24: targeted data, merge and sandbox vote (cached judge, eval/cached/)

| Model | Acc | d0 | d1 | d2 |
|---|---|---|---|---|
| teacher Qwen3-Coder-30B (ceiling) | 57.7 | .76 | .54 | .43 |
| qwen05-v45 + vote (CLI default) | 53.0 | .70 | .50 | .39 |
| qwen05-v4 + vote | 51.3 | .62 | .51 | .41 |
| qwen05-v4 | 50.7 | .59 | .53 | .40 |
| qwen05-v45 (v4/v5 weight average) | 50.0 | .64 | .48 | .38 |
| qwen05-v5 | 49.7 | .67 | .46 | .36 |
| qwen05-v2 | 44.3 | .58 | .38 | .37 |

- Targeted data (scripts/targeted/gen.py): the local teacher writes requests grounded in the sandbox fixture for
  v2's error types (filename used as text, /home/user/ prefixes, commands that never exit, multi-condition find),
  answers each one blind, and a pair is kept only when both commands print the same non-empty output (or are
  file-changing commands that agree). v4 = v2 recipe from base on v2 data + 11k such pairs; v5 = 16.7k pairs.
- DPO, judge DPO, GRPO and hard-case SFT on v2 all stayed at 43-46%.
- Vote (eval/vote.py, engine.rank_by_output): greedy + 5 samples at T=0.7; answers that print the same output in
  the sandbox agree; ties go to greedy. The CLI runs only safe-rated candidates, in one read-only container.

### Sandbox fallback (2026-10-24)
Can the sandbox tell when the voted answer is wrong? `eval/fail_signal.py qwen05-v45`: of 141 wrong answers 46% show
an error or empty output, but so do 43% of the 159 right ones, because the fixture is not the test machine (missing
setup_nl2b_fs_1.sh, no network, no sudo). Only bad flag / missing tool / syntax errors separate them (9 wrong vs 1 right).
So no multi-turn repair loop; instead a broken best answer loses to the next answer that ran cleanly
(`engine.broken`, `vote.py pick --fallback`). Broken samples also lose their vote, as in the eval.
v45 + vote + fallback: **53.7%** (d0 .70, d1 .51, d2 .40); 7 answers changed, 2 fixed, 0 broken. Now the CLI default.
