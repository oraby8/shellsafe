import pytest

from shellsafe import daemon
from shellsafe.engine import Suggestion, default_model
from shellsafe.retrieve import Page
from shellsafe.safety import Risk, Verdict


def test_wire_roundtrip():
    s = [Suggestion("rm -r build", 3, Verdict(Risk.CAUTION, ["deletes files"]), [("rm -r", "recursive")], ["w"])]
    p = [Page("rm", "remove files", [("Remove a file", "rm file")])]
    assert daemon.from_wire(daemon.to_wire(s, p)) == (s, p)


def test_no_daemon_means_none(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "SOCK", tmp_path / "missing.sock")
    assert daemon.status() is None
    assert daemon.stop() is False


MODEL = default_model(download=False)


@pytest.mark.skipif(MODEL is None, reason="needs the model (run `shellsafe ask` once to download it)")
def test_start_ask_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon, "HOME", tmp_path)
    monkeypatch.setattr(daemon, "SOCK", tmp_path / "daemon.sock")
    monkeypatch.setattr(daemon, "LOG", tmp_path / "daemon.log")
    monkeypatch.setenv("SHELLSAFE_HOME", str(tmp_path))  # the child process reads it at import
    daemon.start(MODEL)
    try:
        assert oct((tmp_path / "daemon.sock").stat().st_mode)[-3:] == "600"
        reply = daemon.call({"op": "ask", "request": "count the lines in notes.txt", "n": 1, "samples": 0,
                             "workdir": str(tmp_path)})
        suggestions, _ = daemon.from_wire(reply)
        assert "wc" in suggestions[0].command
    finally:
        assert daemon.stop()
    assert daemon.status() is None
