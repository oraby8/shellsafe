"""Answer the NL2SH-ALFA test prompts with a Hugging Face model in transformers, for models llama.cpp cannot load
or that need a special prompt format. Accuracy only: latency here is GPU time, not comparable to generate.py's CPU.

Usage: python eval/hf_generate.py models/base/<name> --mode chat|tools|fewshot
  chat     eval/models.json default system prompt + the request (as generate.py)
  tools    function calling with one run_bash(command) tool (FunctionGemma's native format)
  fewshot  base models: 8 solved examples from the NL2SH-ALFA *train* split, then "Request: ...\nCommand:"
Writes eval/results/<name>-hf-<mode>.jsonl for eval/score.py.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from generate import ROOT, clean  # noqa: E402

RUN_BASH = {"type": "function", "function": {
    "name": "run_bash", "description": "Run one Bash command on Ubuntu Linux that does what the user asked.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "the command"}},
                   "required": ["command"]}}}


def tool_command(text: str) -> str:
    m = re.search(r"command:<escape>(.*?)<escape>", text, flags=re.S) or re.search(r"command:\s*\"(.*?)\"", text)
    return m.group(1).strip() if m else clean(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--mode", choices=["chat", "tools", "fewshot"], default="chat")
    args = ap.parse_args()

    name = Path(args.model_dir).name
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=args.mode == "fewshot")
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=args.mode == "fewshot").cuda().eval()
    prompt = json.loads((ROOT / "eval/models.json").read_text())["default"]
    shots = pd.read_csv(ROOT / "data/raw/NL2SH-ALFA/train.csv").dropna().sample(8, random_state=0)
    prefix = "".join(f"Request: {r.nl}\nCommand: {r.bash}\n\n" for r in shots.itertuples())
    test = pd.read_csv(ROOT / "data/raw/NL2SH-ALFA/test.csv")
    out = ROOT / f"eval/results/{name}-hf-{args.mode}.jsonl"

    with out.open("w") as f:
        for i, row in test.iterrows():
            if args.mode == "fewshot":
                ids = tok(prefix + f"Request: {row.nl}\nCommand:", return_tensors="pt").input_ids
            else:
                messages = ([{"role": "developer", "content": "You are a model that can do function calling with the "
                                                               "following functions"},
                             {"role": "user", "content": row.nl}] if args.mode == "tools" else
                            [{"role": "system", "content": prompt["system"]},
                             {"role": "user", "content": prompt["user_tmpl"].format(nl=row.nl)}])
                ids = tok.apply_chat_template(messages, tools=[RUN_BASH] if args.mode == "tools" else None,
                                              add_generation_prompt=True, return_tensors="pt", return_dict=True
                                              )["input_ids"]
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(ids.cuda(), max_new_tokens=128, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            raw = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=args.mode != "tools")
            cmd = tool_command(raw) if args.mode == "tools" else clean(raw)
            f.write(json.dumps({"index": int(i), "nl": row.nl, "gold": row.bash, "gold2": row.bash2,
                                "difficulty": int(row.difficulty), "raw": raw, "cmd": cmd,
                                "latency_s": round(time.perf_counter() - t0, 3), "prompt_tokens": int(ids.shape[1]),
                                "completion_tokens": int(gen.shape[1] - ids.shape[1])}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
