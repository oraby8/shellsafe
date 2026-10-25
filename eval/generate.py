"""Generate one command per NL2SH-ALFA test prompt with a GGUF model on CPU via llama-server.

Usage: python eval/generate.py <model-name> [--quant Q8_0] [--threads 8]
       python eval/generate.py <served-model-name> --endpoint http://127.0.0.1:8100/v1   (an already running server)
Writes eval/results/<model>-<quant>.jsonl with the command and per-request latency.
"""
import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests

import sys
ROOT = Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "src"))
from shellsafe import engine  # noqa: E402
from shellsafe.retrieve import ExampleIndex, Retriever, TldrIndex  # noqa: E402

SERVER = ROOT / "third_party/llama.cpp/build-cpu/bin/llama-server"
PORT = 8091


def clean(text: str) -> str:
    """Reduce a model reply to a single command line."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:bash|sh|shell)?\s*\n?(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    line = text.splitlines()[0].strip() if text else ""
    return line.removeprefix("$ ").strip("`").strip()


def build_retriever(test: pd.DataFrame) -> Retriever:
    train = pd.read_csv(ROOT / "data/raw/NL2SH-ALFA/train.csv").dropna()
    # Drop train rows whose request text appears verbatim in the test set.
    leaked = set(test.nl.str.lower().str.strip())
    train = train[~train.nl.str.lower().str.strip().isin(leaked)]
    return Retriever(TldrIndex(), ExampleIndex(list(zip(train.nl, train.bash))))


def build_engine_retriever() -> Retriever:
    """Retriever over the fine-tuning pairs (already decontaminated against the test set), as the CLI ships."""
    rows = [json.loads(l) for l in open(ROOT / "data/stage/final.jsonl")]
    return Retriever(TldrIndex(), ExampleIndex([(r["nl"], r["cmd"]) for r in rows]))


def engine_user(nl: str, retriever: Retriever | None, mode: str) -> str:
    """The exact prompt format the fine-tunes were trained on (scripts/data/format_sft.py)."""
    pages = retriever.commands(nl, 3) if mode in ("both", "docs") else []
    shots = retriever.examples.search(nl, 5) if mode in ("both", "examples") else []
    return engine.build_prompt(nl, pages, shots)


def with_context(nl: str, request: str, retriever: Retriever | None, mode: str, pages_k=3, shots_k=5) -> str:
    """Prepend tldr pages and/or similar solved examples to the (already templated) request."""
    if mode == "none":
        return request
    parts = []
    if mode in ("docs", "both"):
        parts.append("Reference docs:\n" + "\n\n".join(p.render() for p in retriever.commands(nl, pages_k)))
    if mode in ("examples", "both"):
        shots = retriever.examples.search(nl, shots_k)
        parts.append("Similar examples:\n" + "\n".join(f"- {nl} => {cmd}" for _, nl, cmd in shots))
    return "\n\n".join(parts) + "\n\nRequest: " + request


def wait_ready(timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return
        except requests.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--quant", default="Q8_0")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--context", choices=["none", "docs", "examples", "both"], default="none")
    ap.add_argument("--endpoint", help="OpenAI-compatible base URL to query instead of starting llama-server")
    ap.add_argument("--engine-prompt", action="store_true",
                    help="use the CLI/fine-tune prompt (engine.SYSTEM + build_prompt) instead of models.json")
    args = ap.parse_args()

    cfg = json.loads((ROOT / "eval/models.json").read_text())
    prompt = {**cfg["default"], **cfg["models"].get(args.model, {})}
    if args.engine_prompt:
        prompt["system"] = engine.SYSTEM
    gguf = ROOT / f"models/gguf/{args.model}/{args.model}-{args.quant}.gguf"
    suffix = "" if args.context == "none" else f"-ctx_{args.context}"
    tag = "endpoint" if args.endpoint else args.quant
    out = ROOT / f"eval/results/{args.model}-{tag}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(ROOT / "data/raw/NL2SH-ALFA/test.csv")
    if args.limit:
        test = test.head(args.limit)
    retriever = None
    if args.context != "none":
        retriever = build_engine_retriever() if args.engine_prompt else build_retriever(test)

    # Pin to the first N cores so latency approximates a laptop, not a 64-core server.
    server = None
    url = f"{args.endpoint}/chat/completions" if args.endpoint else f"http://127.0.0.1:{PORT}/v1/chat/completions"
    if not args.endpoint:
        server = subprocess.Popen(
            ["taskset", "-c", f"0-{args.threads - 1}", str(SERVER), "-m", str(gguf),
             "--port", str(PORT), "-t", str(args.threads), "-c", "4096", "--jinja"],
            stdout=subprocess.DEVNULL, stderr=open(ROOT / "logs/llama-server.log", "w"))
    try:
        if server:
            wait_ready()
        with out.open("w") as f:
            for i, row in test.iterrows():
                t0 = time.perf_counter()
                if args.engine_prompt:
                    user = engine_user(row.nl, retriever, args.context)
                else:
                    user = with_context(row.nl, prompt["user_tmpl"].format(nl=row.nl), retriever, args.context)
                t_retrieve = time.perf_counter() - t0
                messages = [{"role": "system", "content": prompt["system"]},
                            {"role": "user", "content": user}]
                r = requests.post(url, json={
                    "model": args.model, "messages": messages, "temperature": 0, "max_tokens": 128,
                    "chat_template_kwargs": {"enable_thinking": False}}, timeout=120).json()
                dt = time.perf_counter() - t0
                raw = r["choices"][0]["message"]["content"] or ""
                f.write(json.dumps({"index": int(i), "nl": row.nl, "gold": row.bash, "gold2": row.bash2,
                                    "difficulty": int(row.difficulty), "raw": raw, "cmd": clean(raw),
                                    "latency_s": round(dt, 3), "retrieve_s": round(t_retrieve, 3),
                                    "prompt_tokens": r["usage"]["prompt_tokens"],
                                    "completion_tokens": r["usage"]["completion_tokens"]}) + "\n")
        print("wrote", out)
    finally:
        if server:
            server.terminate()
            server.wait()


if __name__ == "__main__":
    main()
