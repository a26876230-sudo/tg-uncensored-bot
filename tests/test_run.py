import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def fake_app(tmp_path, monkeypatch):
    monkeypatch.delenv("CHAT_BOT_TOKEN", raising=False)
    app = tmp_path / "app"
    app.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "run.py", app / "run.py")
    child_script = """
import os
import signal
import sys
import time
from pathlib import Path

is_bot = Path(__file__).stem == "bot"

def marker_path(name):
    return Path(f"bot-{name}" if is_bot else name)

with marker_path("starts").open("a") as starts:
    starts.write("started\\n")

if os.environ["BOT_MODE" if is_bot else "CHILD_MODE"] == "restart":
    marker = marker_path("attempted")
    if not marker.exists():
        marker.touch()
        sys.exit(17)
    marker_path("restarted").touch()
else:
    def stop(signum, _frame):
        marker_path("received").write_text(str(signum))
        sys.exit(17)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.alarm(10)
    marker_path("ready").touch()
    while True:
        time.sleep(1)
"""
    for name in ("bridge.py", "bot.py"):
        (app / name).write_text(child_script, encoding="utf-8")
    return app


def test_supervisor_restarts_after_failure_then_exits_on_success(fake_app):
    result = subprocess.run(
        [sys.executable, "-c", "import run; run.INITIAL_DELAY = 0.05; raise SystemExit(run.main())"],
        cwd=fake_app.parent,
        env={**os.environ, "PYTHONPATH": str(fake_app), "CHILD_MODE": "restart"},
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert (fake_app / "restarted").exists()
    assert result.stderr.count("exited with code 17; restarting in 0.05 s") == 1
    assert not (fake_app / "bot-starts").exists()


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX signal delivery")
@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("chat_enabled", [False, True])
def test_supervisor_forwards_stop_and_does_not_restart(fake_app, stop_signal, chat_enabled):
    env = {**os.environ, "CHILD_MODE": "stop", "BOT_MODE": "stop"}
    if chat_enabled:
        env["CHAT_BOT_TOKEN"] = "123:fake"
    process = subprocess.Popen(
        [sys.executable, str(fake_app / "run.py")],
        cwd=fake_app.parent,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        ready = [fake_app / "ready"]
        if chat_enabled:
            ready.append(fake_app / "bot-ready")
        deadline = time.monotonic() + 5
        while not all(path.exists() for path in ready) and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert all(path.exists() for path in ready)

        process.send_signal(stop_signal)
        _, stderr = process.communicate(timeout=5)

        assert process.returncode == 0, stderr
        assert (fake_app / "received").read_text() == str(stop_signal)
        if chat_enabled:
            assert (fake_app / "bot-received").read_text() == str(stop_signal)
        assert "restarting" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX signal delivery")
@pytest.mark.parametrize("crashed_child", ["bridge", "bot"])
def test_supervisor_restarts_only_the_failed_child(fake_app, crashed_child):
    process = subprocess.Popen(
        [sys.executable, "-c", "import run; run.INITIAL_DELAY = 0.05; raise SystemExit(run.main())"],
        cwd=fake_app.parent,
        env={
            **os.environ, "PYTHONPATH": str(fake_app), "CHAT_BOT_TOKEN": "123:fake",
            "CHILD_MODE": "restart" if crashed_child == "bridge" else "stop",
            "BOT_MODE": "restart" if crashed_child == "bot" else "stop",
        },
        stderr=subprocess.PIPE,
        text=True,
    )
    crashed_prefix = "bot-" if crashed_child == "bot" else ""
    other_prefix = "" if crashed_child == "bot" else "bot-"
    try:
        restarted = fake_app / f"{crashed_prefix}restarted"
        other_ready = fake_app / f"{other_prefix}ready"
        deadline = time.monotonic() + 5
        while not (restarted.exists() and other_ready.exists()) and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert restarted.exists()
        assert other_ready.exists()

        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=5)

        assert process.returncode == 0, stderr
        assert (fake_app / f"{crashed_prefix}starts").read_text().splitlines() == ["started"] * 2
        assert (fake_app / f"{other_prefix}starts").read_text().splitlines() == ["started"]
        assert (fake_app / f"{other_prefix}received").read_text() == str(signal.SIGTERM)
        assert stderr.count("restarting") == 1
        assert f"{crashed_child}.py exited with code 17; restarting in 0.05 s" in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()


def test_supervisor_does_not_restart_chatbot_without_a_key(fake_app):
    root = Path(__file__).resolve().parents[1]
    for name in ("bot.py", "telegram_common.py"):
        shutil.copyfile(root / name, fake_app / name)
    result = subprocess.run(
        [sys.executable, "-c", "import run; run.INITIAL_DELAY = 0.05; raise SystemExit(run.main())"],
        cwd=fake_app.parent,
        env={
            **os.environ, "PYTHONPATH": str(fake_app), "CHILD_MODE": "restart",
            "CHAT_BOT_TOKEN": "123:fake", "UNLEASHED_API_KEY": "",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert (fake_app / "restarted").exists()
    assert result.stderr.count("Чат-бот не запущен: не задан UNLEASHED_API_KEY.") == 1
    assert "bot.py exited" not in result.stderr
