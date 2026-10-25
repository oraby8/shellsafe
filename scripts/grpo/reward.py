"""Execution + judge reward for GRPO on shell commands.

score(rows) takes dicts with id, request, gold, gold_v (gold's sandbox verdict) and cmd, and returns one reward each:
  1.0  same command as gold, or same non-empty output as gold, or the API judge says it does the job
  0.2  it runs but the judge says it is wrong
  0.0  broken (syntax, bad flags, crashed the sandbox) or an invented tool
  0.5  the judge could not answer (neutral)
minus 0.2 when it is much longer than gold, so the policy cannot pad its way to a "correct" verdict.
Sandbox results and verdicts are cached in memory and on disk (data/grpo/cache.jsonl), so repeats are free.
"""
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

_ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(_ROOT / "scripts/data"))
sys.path.insert(0, str(_ROOT / "scripts/dpo"))
from judge_pairs import SYSTEM as JUDGE_SYSTEM, clip  # noqa: E402
from pairs import KNOWN_TOOLS  # noqa: E402
from verify import DOCKER  # noqa: E402

ENV = dict(l.strip().split("=", 1) for l in open(str(_ROOT / ".llmx.env")) if "=" in l)
MODELS = ["moonshotai/Kimi-K3", "deepseek-ai/DeepSeek-V4.1-Flash", "Qwen/Qwen3.8-Flash-Next"]
CACHE = (_ROOT / "data/grpo/cache.jsonl")


def norm(cmd):
    return " ".join(cmd.split())


class Sandbox:
    """One long-lived container running sandbox/worker.py; restarted if a command kills it."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None

    def run(self, cmd):
        with self.lock:
            for _ in range(2):
                if self.proc is None or self.proc.poll() is not None:
                    self.proc = subprocess.Popen(DOCKER, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                 stderr=subprocess.DEVNULL, text=True, bufsize=1)
                try:
                    self.proc.stdin.write(json.dumps({"id": "x", "cmd": cmd}) + "\n")
                    self.proc.stdin.flush()
                    line = self.proc.stdout.readline()
                except (BrokenPipeError, OSError):
                    line = ""
                if line:
                    res = json.loads(line)
                    res.pop("id", None)
                    return res
                self.proc.kill()
                self.proc = None
                return {"status": "crashed_sandbox"}


class Reward:
    def __init__(self, sandboxes=32, judge_threads=96):
        self.boxes = [Sandbox() for _ in range(sandboxes)]
        self.pool = ThreadPoolExecutor(max(sandboxes, judge_threads))
        self.client = OpenAI(base_url=ENV["LLMX_BASE_URL"], api_key=ENV["LLMX_API_KEY"], timeout=120)
        self.judge_sems = {m: threading.Semaphore(judge_threads // len(MODELS)) for m in MODELS}
        self.exec_cache, self.verdicts = {}, {}
        self.lock = threading.Lock()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        if CACHE.exists():
            for e in map(json.loads, CACHE.open()):
                (self.exec_cache if e["kind"] == "exec" else self.verdicts)[e["key"]] = e["value"]
        self.cache_f = CACHE.open("a")
        self.n = 0

    def _save(self, kind, key, value):
        with self.lock:
            (self.exec_cache if kind == "exec" else self.verdicts)[key] = value
            self.cache_f.write(json.dumps({"kind": kind, "key": key, "value": value}) + "\n")
            self.cache_f.flush()

    def execute(self, cmds):
        todo = sorted({norm(c) for c in cmds} - self.exec_cache.keys())
        futs = [self.pool.submit(self.boxes[i % len(self.boxes)].run, c) for i, c in enumerate(todo)]
        for c, f in zip(todo, futs):
            self._save("exec", c, f.result())
        return [self.exec_cache[norm(c)] for c in cmds]

    def judge(self, row, v):
        key = f"{row['id']}|{norm(row['cmd'])}"
        if key in self.verdicts:
            return self.verdicts[key]
        with self.lock:
            self.n += 1
            model = MODELS[self.n % len(MODELS)]
        msg = (f"{row['request']}\n\nReference: {row['gold']}\n{clip(row['gold_v'])}\n\n"
               f"Candidate: {row['cmd']}\n{clip(v)}")
        verdict = "error"
        with self.judge_sems[model]:
            for _ in range(2):
                try:
                    r = self.client.chat.completions.create(
                        model=model, temperature=0, max_completion_tokens=4000,
                        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": msg}])
                    text = (r.choices[0].message.content or "").strip()
                    got = json.loads(text[text.find("{"):text.rfind("}") + 1]).get("verdict")
                    if got in ("correct", "incorrect"):
                        verdict = got
                        break
                except Exception:
                    continue
        if verdict != "error":
            self._save("verdict", key, verdict)
        return verdict

    def base_score(self, row, v):
        g = row["gold_v"]
        if norm(row["cmd"]) == norm(row["gold"]):
            return 1.0
        if v["status"] in ("syntax", "bad_usage", "crashed_sandbox", "worker_error"):
            return 0.0
        if v["status"] == "missing_tool" and g["status"] != "missing_tool" and v.get("tool") not in KNOWN_TOOLS:
            return 0.0
        if v["status"] == g["status"] == "ok" and v.get("stdout") == g.get("stdout") and g.get("stdout", "").strip():
            return 1.0
        return {"correct": 1.0, "incorrect": 0.2, "error": 0.5}[self.judge(row, v)]

    def score(self, rows):
        verdicts = self.execute([r["cmd"] for r in rows])
        scores = list(self.pool.map(lambda rv: self.base_score(*rv), zip(rows, verdicts)))
        for i, r in enumerate(rows):
            if len(r["cmd"]) > 2 * len(r["gold"]) + 40:
                scores[i] -= 0.2
        return scores
