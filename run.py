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
LOG = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig()
    app_dir = Path(__file__).resolve().parent
    child = None
    stop_signal = None
    forwarded = False

    def forward_stop():
        nonlocal forwarded
        if child is not None and stop_signal is not None and not forwarded:
            forwarded = True
            child.send_signal(stop_signal)

    def request_stop(signum, _frame):
        nonlocal stop_signal
        if stop_signal is None:
            stop_signal = signum
            forward_stop()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    delay = INITIAL_DELAY
    while stop_signal is None:
        started = time.monotonic()
        child = subprocess.Popen(
            [sys.executable, "-u", "bridge.py"],
            cwd=app_dir,
            start_new_session=os.name == "posix",
        )
        forward_stop()
        status = child.wait()
        child = None
        if stop_signal is not None or status == 0:
            break
        if time.monotonic() - started >= RESET_AFTER:
            delay = INITIAL_DELAY
        LOG.warning("bridge.py exited with code %s; restarting in %s s", status, delay)
        retry_at = time.monotonic() + delay
        while stop_signal is None and (remaining := retry_at - time.monotonic()) > 0:
            time.sleep(min(remaining, 0.2))
        delay = min(delay * 2, MAX_DELAY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
