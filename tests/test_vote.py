import pytest

from shellsafe import sandbox
from shellsafe.engine import Suggester, broken, rank_by_output
from shellsafe.sandbox import RunResult


def ok(out):
    return RunResult(True, 0, out)


def test_same_output_outvotes_greedy():
    cands = ["wc -l a.txt | cut -d' ' -f1", "wc -l < a.txt", "cat a.txt | wc -l", "wc -w a.txt"]
    ranked = rank_by_output(cands, [ok("5\n"), ok("5\n"), ok("5"), ok("12 a.txt")])
    assert ranked[0] == ("wc -l a.txt | cut -d' ' -f1", 3)  # greedy represents its group
    assert ranked[1] == ("wc -w a.txt", 1)


def test_majority_beats_greedy():
    cands = ["ls -R", "find . -type f", "find . -type f", "find . -type f -print"]
    ranked = rank_by_output(cands, [ok("x\ny"), ok("./a"), ok("./a"), ok("./a")])
    assert ranked[0] == ("find . -type f", 3)


def test_tie_goes_to_greedy():
    ranked = rank_by_output(["date", "date -u"], [ok("Mon"), ok("Tue")])
    assert ranked[0][0] == "date"


def test_broken_samples_get_no_vote_but_greedy_stays():
    nf = RunResult(False, 127, "bash: frob: command not found")
    ranked = rank_by_output(["frob x", "frob y", "frob y"], [nf, nf, nf])
    assert ranked == [("frob x", 1)]


def test_unrun_and_empty_output_vote_by_text():
    cands = ["rm -rf build", "rm -rf  build", "rm -r build"]
    ranked = rank_by_output(cands, [None, None, None])
    assert ranked[0] == ("rm -rf build", 2)


def test_broken_detects_command_errors_only():
    assert broken(RunResult(False, 127, "bash: frob: command not found"))
    assert "invalid option" in broken(RunResult(False, 1, "ls: invalid option -- 'Q'\nTry 'ls --help'"))
    assert broken(RunResult(False, 2, "bash: -c: line 1: syntax error near unexpected token `|'"))
    assert not broken(RunResult(False, 1, "cat: a.txt: No such file or directory"))  # depends on the folder
    assert not broken(RunResult(True, 0, "usage: shown by a working command"))
    assert not broken(None)


def test_bad_flag_samples_get_no_vote():
    bad = RunResult(False, 1, "ls: invalid option -- 'Q'")
    ranked = rank_by_output(["ls -a", "ls -Q", "ls -Q"], [ok(".\n.."), bad, bad])
    assert ranked == [("ls -a", 1)]


class FakeSuggester(Suggester):
    """Suggester with canned answers and sandbox results, so suggest() runs without a model or Docker."""

    def __init__(self, cands, results):
        self.cands, self.results, self.has_sandbox = cands, results, True
        self.retriever = type("R", (), {"commands": lambda self, q, k: [],
                                        "examples": type("E", (), {"search": lambda self, q, k: []})()})()

    def _generate(self, prompt, temperature, seed):
        return self.cands[seed]


def test_broken_best_answer_falls_back(monkeypatch):
    cands = ["ls -Q", "ls -Q", "ls -Q", "ls -a", "ls -A", "ls -a"]
    bad = RunResult(False, 1, "ls: invalid option -- 'Q'\nTry 'ls --help' for more information.")
    s = FakeSuggester(cands, None)
    monkeypatch.setattr(sandbox, "run_many", lambda cmds, wd: [bad if c == "ls -Q" else ok(".\n..") for c in cmds])
    out, _ = s.suggest("show hidden files", samples=5, workdir="/tmp")
    assert out[0].command == "ls -a"  # the samples outvote the broken greedy answer...
    s = FakeSuggester(["ls -Q", "ls -Q", "ls -Q", "ls -Q", "ls -a", "ls -Q"], None)
    out, _ = s.suggest("show hidden files", samples=5, workdir="/tmp")
    assert out[0].command == "ls -a"  # ...and a lone working sample beats a broken best answer
    assert out[1].command == "ls -Q" and any("invalid option" in w for w in out[1].warnings)


def test_broken_answer_kept_when_nothing_else_runs(monkeypatch):
    s = FakeSuggester(["ls -Q"] * 6, None)
    monkeypatch.setattr(sandbox, "run_many", lambda cmds, wd: [RunResult(False, 1, "ls: invalid option -- 'Q'")])
    out, _ = s.suggest("show hidden files", samples=5, workdir="/tmp")
    assert out[0].command == "ls -Q" and any("invalid option" in w for w in out[0].warnings)


@pytest.mark.skipif(not sandbox.available(), reason="needs Docker")
def test_run_many_splits_outputs(tmp_path):
    (tmp_path / "a.txt").write_text("one\ntwo\n")
    r = sandbox.run_many(["wc -l < a.txt", "false", "sleep 30", "printf 'no newline'"], str(tmp_path), timeout=2)
    assert r[0].ok and r[0].output.strip() == "2"
    assert not r[1].ok and r[1].exit_code == 1
    assert r[2].note.startswith("timed out")
    assert r[3].output == "no newline"
