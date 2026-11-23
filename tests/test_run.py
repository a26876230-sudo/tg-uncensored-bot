import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def fake_app(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "run.py", app / "run.py")
    (app / "bridge.py").write_text(
        """
import os
import signal
import sys
import time
from pathlib import Path

if os.environ["CHILD_MODE"] == "restart":
    marker = Path("attempted")
    if not marker.exists():
        marker.touch()
        sys.exit(17)
    Path("restarted").touch()
else:
    def stop(signum, _frame):
        Path("received").write_text(str(signum))
        sys.exit(17)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.alarm(10)
    Path("ready").touch()
    while True:
        time.sleep(1)
""",
        encoding="utf-8",
    )
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


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX signal delivery")
@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_supervisor_forwards_stop_and_does_not_restart(fake_app, stop_signal):
    process = subprocess.Popen(
        [sys.executable, str(fake_app / "run.py")],
        cwd=fake_app.parent,
        env={**os.environ, "CHILD_MODE": "stop"},
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (fake_app / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (fake_app / "ready").exists()

        process.send_signal(stop_signal)
        _, stderr = process.communicate(timeout=5)

        assert process.returncode == 0, stderr
        assert (fake_app / "received").read_text() == str(stop_signal)
        assert "restarting" not in stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()
