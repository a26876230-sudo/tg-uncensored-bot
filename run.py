import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


INITIAL_DELAY = 5
MAX_DELAY = 60
RESET_AFTER = 600
POLL_INTERVAL = 0.2
LOG = logging.getLogger(__name__)


class Child:
    def __init__(self, script: str):
        self.script = script
        self.process = None
        self.started_at = 0.0
        self.retry_at = 0.0
        self.delay = INITIAL_DELAY

    def tick(self, app_dir: Path) -> bool:
        now = time.monotonic()
        if self.process is None:
            if now >= self.retry_at:
                self.started_at = now
                self.process = subprocess.Popen(
                    [sys.executable, "-u", self.script],
                    cwd=app_dir,
                    start_new_session=os.name == "posix",
                )
            return True

        status = self.process.poll()
        if status is None:
            return True
        self.process = None
        if status == 0:
            return False
        if now - self.started_at >= RESET_AFTER:
            self.delay = INITIAL_DELAY
        LOG.warning("%s exited with code %s; restarting in %s s", self.script, status, self.delay)
        self.retry_at = now + self.delay
        self.delay = min(self.delay * 2, MAX_DELAY)
        return True


def main() -> int:
    logging.basicConfig()
    app_dir = Path(__file__).resolve().parent
    children = [Child("bridge.py")]
    if os.environ.get("CHAT_BOT_TOKEN", "").strip():
        children.append(Child("bot.py"))
    stop_signal = None

    def request_stop(signum, _frame):
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    while children and stop_signal is None:
        for child in children[:]:
            if stop_signal is not None:
                break
            if not child.tick(app_dir):
                children.remove(child)
        if children and stop_signal is None:
            now = time.monotonic()
            next_tick = min(
                (child.retry_at for child in children if child.process is None),
                default=now + POLL_INTERVAL,
            )
            time.sleep(max(0, min(next_tick - now, POLL_INTERVAL)))

    if stop_signal is not None:
        processes = [child.process for child in children if child.process is not None]
        for process in processes:
            process.send_signal(stop_signal)
        for process in processes:
            process.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
