"""Optional headless Chrome --dump-dom fetcher (scout-preferred CF path)."""
from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
import time

CHROME = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")


class ChromeBlocked(RuntimeError):
    pass


def chrome_dump(url: str, timeout: int = 120) -> str:
    if not CHROME:
        raise RuntimeError("Chrome/Chromium not installed")
    with tempfile.TemporaryDirectory(prefix="tibia-chrome-") as tmp:
        cmd = [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            f"--user-data-dir={tmp}",
            "--virtual-time-budget=20000",
            "--dump-dom",
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        html = proc.stdout or ""
        if not html and proc.returncode != 0:
            raise RuntimeError(f"chrome exit {proc.returncode}: {(proc.stderr or '')[-400:]}")
        head = html[:1000]
        if "429 Too Many Requests" in html or "Too Many Requests" in head:
            raise ChromeBlocked("chrome got HTTP 429")
        if "Just a moment" in head:
            raise ChromeBlocked("chrome still hit Cloudflare challenge")
        if "you have been blocked" in head.lower():
            raise ChromeBlocked("chrome WAF block page")
        return html


class ChromeSession:
    def __init__(self, min_delay: float = 3.5, max_delay: float = 6.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last = 0.0

    def get(self, url: str, timeout: int = 120) -> str:
        elapsed = time.monotonic() - self._last
        wait = random.uniform(self.min_delay, self.max_delay) - elapsed
        if wait > 0:
            time.sleep(wait)
        html = chrome_dump(url, timeout=timeout)
        self._last = time.monotonic()
        return html
