import json
import threading
import time
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "mcp-fs"))
import server  # noqa: E402


@pytest.fixture()
def fs_root(tmp_path):
    original_root = server.FS_ROOT
    original_write = server.ENABLE_WRITE
    server.FS_ROOT = tmp_path
    server.ENABLE_WRITE = True
    yield tmp_path
    server.FS_ROOT = original_root
    server.ENABLE_WRITE = original_write


def test_write_file_append_atomic(fs_root):
    result = server.write_file(path="data.txt", content="hello", atomic=True)
    assert result["created"] is True
    assert fs_root.joinpath("data.txt").read_text() == "hello"

    result = server.write_file(path="data.txt", content=" world", append=True, atomic=True)
    assert result["created"] is False
    assert fs_root.joinpath("data.txt").read_text() == "hello world"


def test_tail_file_follow(fs_root):
    log_path = fs_root / "logs" / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("")

    def _writer():
        time.sleep(0.1)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("update\n")

    thread = threading.Thread(target=_writer, daemon=True)
    thread.start()
    result = server.tail_file(path=str(log_path.relative_to(fs_root)), tail_lines=0, follow_ms=1000, max_bytes=1024)
    thread.join()
    assert "update" in result["content"]["text"]


def test_jsonl_append_atomic(fs_root):
    record_a = {"who": "agent", "goal": "test", "observed": "ok"}
    record_b = {"who": "human", "goal": "review", "observed": "pending"}

    out_a = server.jsonl_append(path="notes/issues.jsonl", record=record_a, create_parents=True)
    out_b = server.jsonl_append(path="notes/issues.jsonl", record=record_b, create_parents=True)

    assert out_a["ok"] and out_b["ok"]

    issues_path = fs_root / "notes" / "issues.jsonl"
    lines = issues_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == record_a
    assert json.loads(lines[1]) == record_b


def test_init_includes_agents_docs(fs_root):
    root_agents = fs_root / "AGENTS.md"
    nested_agents = fs_root / "nested" / "AGENTS.md"

    root_agents.write_text("root instructions")
    nested_agents.parent.mkdir(parents=True, exist_ok=True)
    nested_agents.write_text("nested instructions")

    result = server.init()

    assert "agents" in result
    assert {"path": "AGENTS.md", "content": "root instructions"} in result["agents"]

    assert any(
        entry["path"] == "nested/AGENTS.md" and entry["content"] == "nested instructions"
        for entry in result["agents"]
    )

