"""BM25 lookup over tldr pages: find the command docs most relevant to a free-text request."""
import os
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

# scripts/setup.sh clones tldr-pages here; SHELLSAFE_TLDR points at another checkout's pages/ folder.
TLDR_ROOT = Path(os.environ.get("SHELLSAFE_TLDR", Path(__file__).resolve().parents[2] / "third_party/tldr/pages"))
PLATFORMS = ("common", "linux")

STOPWORDS = set("""a an and are as at be by for from in into is it its of on or that the this to with
my your me i you which what how do does""".split())


@dataclass
class Page:
    name: str
    description: str
    examples: list[tuple[str, str]]  # (what it does, command)

    def render(self, max_examples: int = 6, query: str | None = None) -> str:
        examples = self.examples
        if query and len(examples) > max_examples:
            # Long pages (e.g. learned tools with hundreds of options): show the options closest to the request.
            q = set(tokenize(query))
            examples = sorted(examples, key=lambda e: -len(q & set(tokenize(e[0] + " " + e[1]))))
        lines = [f"{self.name}: {self.description}"]
        lines += [f"- {what}: {cmd}" for what, cmd in examples[:max_examples]]
        return "\n".join(lines)


def _clean_placeholders(cmd: str) -> str:
    # tldr writes options as {{[-C|--directory]}} and arguments as {{path/to/file}}.
    cmd = re.sub(r"\{\{\[([^|\]]+)\|[^\]]*\]\}\}", r"\1", cmd)
    return re.sub(r"\{\{(.*?)\}\}", r"<\1>", cmd)


def parse_page(path: Path) -> Page:
    text = path.read_text(encoding="utf-8")
    desc = " ".join(l[2:].strip() for l in text.splitlines()
                    if l.startswith("> ") and not l.startswith("> More information"))
    examples = []
    for what, cmd in re.findall(r"^- (.+?):?\s*\n\s*\n`(.+)`", text, flags=re.M):
        examples.append((re.sub(r"\[(\w+)\]", r"\1", what), _clean_placeholders(cmd)))
    return Page(name=path.stem, description=desc, examples=examples)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower()) if t not in STOPWORDS]


class TldrIndex:
    def __init__(self, root: Path = TLDR_ROOT, platforms=PLATFORMS, extra_dirs: tuple[Path, ...] = ()):
        pages: dict[str, Page] = {}
        # Later directories win: linux overrides common, and pages the user taught override both.
        for d in [root / p for p in platforms] + list(extra_dirs):
            for path in sorted(d.glob("*.md")):
                pages[path.stem] = parse_page(path)
        self.pages = list(pages.values())
        self.by_name = {p.name: p for p in self.pages}
        docs = []
        for p in self.pages:
            # Repeat the command name so a request that names the tool ranks it first.
            name_tokens = tokenize(p.name.replace("-", " ")) * 3 + [p.name]
            docs.append(name_tokens + tokenize(p.description) + tokenize(" ".join(w for w, _ in p.examples)))
        self.bm25 = BM25Okapi(docs)

    def search(self, query: str, k: int = 3) -> list[Page]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        hits = [self.pages[i] for i in top if scores[i] > 0]
        # A request that literally names a command should always include that page.
        named = [self.by_name[t] for t in tokens if t in self.by_name and self.by_name[t] not in hits]
        return (named + hits)[:k]


SKIP_NAMES = {"sudo", "xargs", "time", "env", "nohup", "then", "do", "if", "for", "while", "true", "false"}


def command_names(cmd: str) -> list[str]:
    """First word of each pipeline/list segment, e.g. 'find . | xargs grep x' -> ['find', 'grep']."""
    names = []
    for seg in re.split(r"\|\||&&|[|;]|\$\(|`|xargs\s+(?:-\S+\s+)*", cmd):
        words = [w for w in seg.strip().split() if not re.match(r"^\w+=", w)]
        if words and words[0] not in SKIP_NAMES and re.match(r"^[a-z][\w.+-]*$", words[0]):
            names.append(words[0])
    return names


class ExampleIndex:
    """BM25 over known (request, command) pairs; used for few-shot examples and to vote for commands."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.pairs = pairs
        self.bm25 = BM25Okapi([tokenize(nl) or ["_"] for nl, _ in pairs])

    def search(self, query: str, k: int = 5) -> list[tuple[float, str, str]]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [(scores[i], *self.pairs[i]) for i in top if scores[i] > 0]


class Retriever:
    """Combine example votes with direct tldr matches to pick the command pages to show the model."""

    def __init__(self, tldr: TldrIndex, examples: ExampleIndex, vote_k: int = 20):
        self.tldr, self.examples, self.vote_k = tldr, examples, vote_k

    def commands(self, query: str, k: int = 3) -> list[Page]:
        votes: dict[str, float] = {}
        for score, _, cmd in self.examples.search(query, self.vote_k):
            for name in set(command_names(cmd)):
                if name in self.tldr.by_name:
                    votes[name] = votes.get(name, 0.0) + score
        # Direct tldr hits break ties and cover tools the examples never use.
        for rank, page in enumerate(self.tldr.search(query, k)):
            votes[page.name] = votes.get(page.name, 0.0) + (k - rank) * 0.5
        best = sorted(votes, key=votes.get, reverse=True)[:k]
        return [self.tldr.by_name[n] for n in best]
