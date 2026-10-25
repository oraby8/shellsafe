"""Time each stage of one suggestion so we know where the latency goes."""
import time

t = time.perf_counter()
from llama_cpp import Llama  # noqa: E402
import shellsafe.engine as E  # noqa: E402
print(f"imports      {time.perf_counter() - t:.2f}s")

t = time.perf_counter()
llm = Llama(model_path=E.default_model(), n_ctx=4096, n_threads=8, n_threads_batch=8, n_batch=1024, flash_attn=True, verbose=False)
print(f"model load   {time.perf_counter() - t:.2f}s")

t = time.perf_counter()
s = E.Suggester.__new__(E.Suggester)
s.llm = llm
import pandas as pd  # noqa: E402
tldr = E.TldrIndex(extra_dirs=(E.docs.LEARNED_DIR,))
train = pd.read_csv(E.EXAMPLES_CSV).dropna()
s.retriever = E.Retriever(tldr, E.ExampleIndex(list(zip(train.nl, train.bash))))
print(f"index build  {time.perf_counter() - t:.2f}s")

q = "find files bigger than 100MB modified in the last week"
t = time.perf_counter()
pages = s.retriever.commands(q, 3)
shots = s.retriever.examples.search(q, 5)
prompt = E.build_prompt(q, pages, shots)
print(f"retrieval    {time.perf_counter() - t:.2f}s  prompt tokens={len(llm.tokenize(prompt.encode()))}")
for label, temp, seed in [("greedy", 0.0, 0), ("sample 1", 0.7, 1), ("sample 2", 0.7, 2)]:
    t = time.perf_counter()
    out = s._generate(prompt, temp, seed)
    print(f"{label:12s} {time.perf_counter() - t:.2f}s  {out}")
t = time.perf_counter()
E.explain_flags(out, {p.name: p for p in pages})
print(f"flag check   {time.perf_counter() - t:.2f}s")
