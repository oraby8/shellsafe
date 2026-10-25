"""request -> lookup -> small model -> checks. The model only writes commands; everything else is deterministic."""
import os
import re
import shlex
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


from shellsafe import docs, sandbox
from shellsafe.retrieve import ExampleIndex, Page, Retriever, TldrIndex, command_names
from shellsafe.safety import Risk, Verdict, assess

ROOT = Path(__file__).resolve().parents[2]  # the repo checkout
HF_MODEL = ("ahmedsamirtarjama/shellsafe-qwen2.5-coder-0.5b", "qwen05-v45-Q8_0.gguf")
# The shellsafe fine-tunes were trained on the bare request; retrieved docs/examples in the prompt made them worse.
# Third-party models such as kitty-bash-llm do better with SHELLSAFE_CONTEXT=1.
USE_CONTEXT = os.environ.get("SHELLSAFE_CONTEXT", "0") == "1"
EXAMPLES_CSV = os.environ.get("SHELLSAFE_EXAMPLES", str(ROOT / "data/raw/NL2SH-ALFA/train.csv"))


def default_model(download: bool = True) -> str | None:
    """SHELLSAFE_MODEL, else a local build in models/, else the published model from the Hugging Face Hub (fetched
    once, about 0.5 GB). With download=False, None when it would have to be fetched."""
    if os.environ.get("SHELLSAFE_MODEL"):
        return os.environ["SHELLSAFE_MODEL"]
    local = ROOT / "models/gguf/qwen05-v45" / HF_MODEL[1]
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import GatedRepoError

    cached = try_to_load_from_cache(*HF_MODEL)
    if isinstance(cached, str):
        return cached
    if not download:
        return None
    try:
        return hf_hub_download(*HF_MODEL)
    except GatedRepoError:
        raise RuntimeError(f"The model needs a one-time click to accept its terms: open https://huggingface.co/"
                           f"{HF_MODEL[0]}, agree, then log in with `hf auth login` (or set HF_TOKEN). Or set "
                           f"SHELLSAFE_MODEL to a local GGUF file.") from None
SYSTEM = ("You are a bash command generator. Given a natural language request, "
          "output only the bash command that accomplishes it. No explanation, no markdown fences.")


@dataclass
class Suggestion:
    command: str
    votes: int
    verdict: Verdict
    flags: list[tuple[str, str]] = field(default_factory=list)  # (flag, what it does)
    warnings: list[str] = field(default_factory=list)


def build_prompt(request: str, pages: list[Page], shots: list[tuple[float, str, str]]) -> str:
    parts = []
    if pages:
        parts.append("Reference docs:\n" + "\n\n".join(p.render(query=request) for p in pages))
    if shots:
        parts.append("Similar examples:\n" + "\n".join(f"- {nl} => {cmd}" for _, nl, cmd in shots))
    parts.append("Request: " + request)
    return "\n\n".join(parts)


def first_command_line(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    fence = re.search(r"```(?:bash|sh|shell)?\s*\n?(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1).strip()
    line = text.splitlines()[0].strip() if text else ""
    return line.removeprefix("$ ").strip("`").strip()


# Same patterns as sandbox/worker.py, which labelled the training data.
BAD_USAGE = re.compile(
    r"invalid option|unrecognized option|unknown option|illegal option|invalid argument|"
    r"option requires an argument|unrecognized argument|unknown argument|no such option|"
    r"invalid (?:number|time|date|predicate|expression)|unknown predicate|paths must precede expression|"
    r"Try '[^']+ --help'|^usage:|syntax error|unexpected (?:token|EOF)", re.I | re.M)
NOT_FOUND = re.compile(r"(?:^|: )([\w.+-]+): (?:command )?not found", re.M)


def broken(r: "sandbox.RunResult | None") -> str:
    """Why a sandbox run shows the command itself is wrong (missing tool, bad flag, syntax), or "" if it does not.
    Other failures, such as a missing file, are not counted: they often depend on the folder, not the command."""
    if r is None or r.exit_code in (0, None):
        return ""
    if r.exit_code in (126, 127) or NOT_FOUND.search(r.output):
        return "the command is not installed or not executable"
    m = BAD_USAGE.search(r.output)
    if m:
        line = next((ln for ln in r.output.splitlines() if m.group(0) in ln), m.group(0))
        return f"it failed in the sandbox: {line.strip()[:120]}"
    return ""


def rank_by_output(cands: list[str], results: list["sandbox.RunResult | None"]) -> list[tuple[str, int]]:
    """Majority vote: answers agree when they print the same non-empty output in the sandbox, else when their text
    matches. cands[0] is the greedy answer: it always gets a vote, wins ties, and represents its group. Samples that
    could not run (broken, timed out) get no vote. results[i] is None when cands[i] was not run. Returns (command,
    votes) per group, best first."""
    groups: dict[tuple, list[tuple[int, str]]] = {}
    for k, (cmd, r) in enumerate(zip(cands, results)):
        if not cmd or (k and r is not None and (r.note or broken(r))):
            continue
        out = r.output.strip() if r is not None and r.ok else ""
        groups.setdefault(("out", out) if out else ("cmd", " ".join(cmd.split())), []).append((k, cmd))
    ranked = sorted(groups.values(), key=lambda g: (len(g), any(k == 0 for k, _ in g)), reverse=True)
    return [(next((c for k, c in g if k == 0), Counter(c for _, c in g).most_common(1)[0][0]), len(g)) for g in ranked]


def explain_flags(command: str, pages: dict[str, Page]) -> tuple[list[tuple[str, str]], list[str]]:
    """Describe each flag from the man page; warn about flags no local doc mentions (likely invented)."""
    explained, warnings = [], []
    for seg in re.split(r"\|\||&&|[|;]", command):
        try:
            argv = shlex.split(seg)
        except ValueError:
            continue
        if not argv or argv[0] in ("sudo", "xargs"):
            argv = argv[1:] if argv else argv
        if not argv:
            continue
        name = argv[0]
        doc = docs.doc_text(name)
        tldr = pages.get(name)
        corpus = doc + "\n" + (tldr.render(max_examples=100) if tldr else "")
        for arg in argv[1:]:
            if not re.match(r"^--?[A-Za-z]", arg):
                continue
            flag = arg.split("=")[0]
            # Expand grouped short flags (-la -> -l, -a), except find-style single-dash words (-name).
            singles = [f"-{c}" for c in flag[1:]] if re.fullmatch(r"-[A-Za-z]{2,4}", flag) and name != "find" else [flag]
            for f in singles:
                desc = docs.flag_description(name, f, doc)
                if desc:
                    explained.append((f"{name} {f}", desc))
                elif corpus.strip() and not re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", corpus):
                    warnings.append(f"`{f}` is not mentioned in the {name} docs; it may not exist")
    return explained, warnings


class Suggester:
    def __init__(self, model_path: str | None = None, threads: int | None = None):
        import pandas as pd  # imported here: the daemon client never needs it
        from llama_cpp import Llama

        model_path = model_path or default_model()
        threads = threads or min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 4)
        # n_threads_batch defaults to every CPU on the box; oversubscribing pinned cores made prompts ~3x slower.
        self.llm = Llama(model_path=model_path, n_ctx=4096, n_threads=threads, n_threads_batch=threads,
                         n_batch=1024, flash_attn=True, verbose=False)
        train = pd.read_csv(EXAMPLES_CSV).dropna() if Path(EXAMPLES_CSV).exists() else pd.DataFrame(columns=["nl", "bash"])
        self.examples = ExampleIndex(list(zip(train.nl, train.bash)) or [("", "")])
        self.reload_docs()
        self.has_sandbox = sandbox.available()

    def reload_docs(self):
        """Re-read tldr pages and tools taught with `shellsafe learn`."""
        self.retriever = Retriever(TldrIndex(extra_dirs=(docs.LEARNED_DIR,)), self.examples)

    def _generate(self, prompt: str, temperature: float, seed: int) -> str:
        r = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=128, seed=seed)
        return first_command_line(r["choices"][0]["message"]["content"] or "")

    def suggest(self, request: str, n: int = 3, samples: int = 5,
                workdir: str | None = None) -> tuple[list[Suggestion], list[Page]]:
        """Greedy answer plus `samples` sampled ones, voted on by their output in a read-only sandbox of `workdir`
        (by text when there is no workdir or Docker). Only commands rated safe are ever run."""
        pages = self.retriever.commands(request, 3)
        shots = self.retriever.examples.search(request, 5) if USE_CONTEXT else []
        prompt = build_prompt(request, pages if USE_CONTEXT else [], shots)
        cands = [self._generate(prompt, 0.0, 0)] + [self._generate(prompt, 0.7, s + 1) for s in range(samples)]
        results: list[sandbox.RunResult | None] = [None] * len(cands)
        if samples and workdir and self.has_sandbox:
            runnable = [c for c in dict.fromkeys(cands) if c and assess(c).risk == Risk.SAFE]
            if runnable:
                by_cmd = dict(zip(runnable, sandbox.run_many(runnable, workdir)))
                results = [by_cmd.get(c) for c in cands]
        ranked = rank_by_output(cands, results)
        why = {c: broken(r) for c, r in zip(cands, results)}
        # A best answer the sandbox showed broken drops behind the next answer that ran without such an error.
        if why.get(ranked[0][0]):
            fix = next((i for i, (c, _) in enumerate(ranked) if i and results[cands.index(c)] and not why[c]), None)
            if fix is not None:
                ranked = [ranked[fix], ranked[0]] + [g for i, g in enumerate(ranked) if i not in (0, fix)]
        # Best group first; alternatives only when at least two answers agree, since one-off samples are often wrong.
        # The demoted answer stays visible, with its error, so the user sees why it lost.
        order = ranked[:1] + [(c, v) for c, v in ranked[1:] if v >= 2 or why.get(c)]
        by_name = {p.name: p for p in pages}
        out = []
        for cmd, votes in order[:n]:
            flags, warnings = explain_flags(cmd, by_name)
            if why.get(cmd):
                warnings.append(why[cmd])
            missing = [c for c in dict.fromkeys(command_names(cmd)) if not shutil.which(c)]
            warnings += [f"`{c}` is not installed on this machine" for c in missing]
            out.append(Suggestion(cmd, votes, assess(cmd), flags, warnings))
        return out, pages
