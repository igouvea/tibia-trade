"""HTTP helpers with Cloudflare bypass, rate limiting, retries, Chrome fallback."""
from __future__ import annotations

import os
import random
import time
from typing import Iterable

import cloudscraper
from tenacity import retry, stop_after_attempt, wait_exponential

from scrape.chrome_fetch import CHROME, chrome_dump

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


def _on_vercel() -> bool:
    return bool(os.environ.get("VERCEL") or os.environ.get("VERCEL_ENV"))


class RateLimitedSession:
    def __init__(
        self,
        min_delay: float = 0.8,
        max_delay: float = 1.6,
        user_agent: str = DEFAULT_UA,
        prefer_chrome: bool = False,
        max_attempts: int | None = None,
    ):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.prefer_chrome = prefer_chrome and bool(CHROME)
        self._last = 0.0
        self.session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        self.session.headers.update({"User-Agent": user_agent})
        self._cf_strikes = 0
        if max_attempts is None:
            max_attempts = 2 if _on_vercel() else 5
        self.max_attempts = max_attempts

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last
        wait = random.uniform(self.min_delay, self.max_delay) - elapsed
        if wait > 0:
            time.sleep(wait)

    def _chrome(self, url: str) -> str:
        html = chrome_dump(url)
        self._last = time.monotonic()
        return html

    def get(self, url: str, timeout: int | None = None) -> str:
        if timeout is None:
            timeout = 20 if _on_vercel() else 90

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8 if _on_vercel() else 60),
            reraise=True,
        )
        def _once() -> str:
            self._throttle()
            if self.prefer_chrome and CHROME:
                return self._chrome(url)

            try:
                resp = self.session.get(url, timeout=timeout)
            except Exception:
                if CHROME and not _on_vercel():
                    time.sleep(random.uniform(3.0, 5.0))
                    return self._chrome(url)
                raise

            self._last = time.monotonic()
            body_head = resp.text[:1000] if resp.text else ""
            blocked = (
                resp.status_code in (429, 503)
                or (resp.status_code == 403)
                or ("Just a moment" in body_head)
                or ("you have been blocked" in body_head.lower())
            )
            if blocked:
                self._cf_strikes += 1
                if CHROME and not _on_vercel():
                    time.sleep(random.uniform(3.5, 6.5))
                    return self._chrome(url)
                raise RuntimeError(f"blocked status={resp.status_code}")
            resp.raise_for_status()
            self._cf_strikes = max(0, self._cf_strikes - 1)
            return resp.text

        return _once()


def auction_list_url(page: int) -> str:
    return (
        "https://www.tibia.com/charactertrade/"
        f"?subtopic=pastcharactertrades&currentpage={page}"
    )


def auction_detail_url(
    auction_id: int, subtopic: str = "currentcharactertrades"
) -> str:
    return (
        "https://www.tibia.com/charactertrade/"
        f"?auctionid={auction_id}&page=details&subtopic={subtopic}"
    )


def auction_detail_urls(auction_id: int) -> list[str]:
    """Prefer current (live) trades, then past (sold) — both often resolve the same page."""
    return [
        auction_detail_url(auction_id, "currentcharactertrades"),
        auction_detail_url(auction_id, "pastcharactertrades"),
    ]


def auction_ajax_url(auction_id: int, type_: str, page: int = 1) -> str:
    return (
        "https://www.tibia.com/websiteservices/handle_charactertrades.php"
        f"?auctionid={auction_id}&type={type_}&currentpage={page}"
    )


def _looks_blocked(html: str, status: int | None = None) -> bool:
    head = (html or "")[:1500]
    if status in (403, 429, 503):
        return True
    low = head.lower()
    return (
        "Just a moment" in head
        or "cf-challenge" in low
        or "you have been blocked" in low
        or "attention required" in low
    )


def fetch_auction_html(auction_id: int, timeout: int = 25) -> tuple[str, str]:
    """Server-side HTML fetch for an auction. Returns (html, source_label).

    Order: optional proxy → curl_cffi → cloudscraper → Chrome (non-Vercel).
    Tries current then past character-trade URLs.
    """
    urls = auction_detail_urls(auction_id)
    errors: list[str] = []

    proxy = (os.environ.get("TIBIA_FETCH_PROXY") or "").rstrip("/")
    if proxy:
        import json as _json
        import urllib.error
        import urllib.request

        for url in urls:
            try:
                req = urllib.request.Request(
                    f"{proxy}/fetch?url={url}",
                    headers={"User-Agent": DEFAULT_UA, "Accept": "text/html"},
                    method="GET",
                )
                token = os.environ.get("TIBIA_FETCH_PROXY_TOKEN")
                if token:
                    req.add_header("Authorization", f"Bearer {token}")
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                    ctype = resp.headers.get("Content-Type", "")
                if "application/json" in ctype:
                    data = _json.loads(body)
                    html = data.get("html") or ""
                else:
                    html = body
                if html and not _looks_blocked(html):
                    return html, "proxy"
                errors.append("proxy: empty or blocked")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"proxy: {exc}")

    # curl_cffi — TLS fingerprint closer to Chrome (works better than plain requests on some hosts)
    try:
        from curl_cffi import requests as crequests

        for url in urls:
            try:
                r = crequests.get(
                    url,
                    impersonate="chrome131",
                    timeout=timeout,
                    headers={"User-Agent": DEFAULT_UA},
                )
                if r.status_code == 200 and r.text and not _looks_blocked(r.text, r.status_code):
                    return r.text, "curl_cffi"
                errors.append(f"curl_cffi: status={r.status_code}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"curl_cffi: {exc}")
    except ImportError:
        errors.append("curl_cffi: not installed")

    # cloudscraper single-shot (no long retry storms on Vercel)
    try:
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        scraper.headers.update({"User-Agent": DEFAULT_UA})
        for url in urls:
            try:
                resp = scraper.get(url, timeout=timeout)
                if (
                    resp.status_code == 200
                    and resp.text
                    and not _looks_blocked(resp.text, resp.status_code)
                ):
                    return resp.text, "cloudscraper"
                errors.append(f"cloudscraper: status={resp.status_code}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"cloudscraper: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cloudscraper: {exc}")

    if CHROME and not _on_vercel():
        for url in urls:
            try:
                html = chrome_dump(url, timeout=min(90, timeout + 60))
                if html and not _looks_blocked(html):
                    return html, "chrome"
                errors.append("chrome: blocked or empty")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"chrome: {exc}")

    raise RuntimeError("; ".join(errors[:6]) or "all fetch strategies failed")
