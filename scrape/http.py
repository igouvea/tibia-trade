"""HTTP helpers with Cloudflare bypass, rate limiting, retries, Chrome fallback."""
from __future__ import annotations

import random
import time

import cloudscraper
from tenacity import retry, stop_after_attempt, wait_exponential

from scrape.chrome_fetch import CHROME, chrome_dump

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)


class RateLimitedSession:
    def __init__(
        self,
        min_delay: float = 0.8,
        max_delay: float = 1.6,
        user_agent: str = DEFAULT_UA,
        prefer_chrome: bool = False,
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

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last
        wait = random.uniform(self.min_delay, self.max_delay) - elapsed
        if wait > 0:
            time.sleep(wait)

    def _chrome(self, url: str) -> str:
        html = chrome_dump(url)
        self._last = time.monotonic()
        return html

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    def get(self, url: str, timeout: int = 90) -> str:
        self._throttle()
        if self.prefer_chrome and CHROME:
            return self._chrome(url)

        try:
            resp = self.session.get(url, timeout=timeout)
        except Exception:
            if CHROME:
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
            if CHROME:
                time.sleep(random.uniform(3.5, 6.5))
                return self._chrome(url)
            raise RuntimeError(f"blocked status={resp.status_code}")
        resp.raise_for_status()
        self._cf_strikes = max(0, self._cf_strikes - 1)
        return resp.text


def auction_list_url(page: int) -> str:
    return (
        "https://www.tibia.com/charactertrade/"
        f"?subtopic=pastcharactertrades&currentpage={page}"
    )


def auction_detail_url(auction_id: int) -> str:
    return (
        "https://www.tibia.com/charactertrade/"
        f"?auctionid={auction_id}&page=details&subtopic=pastcharactertrades"
    )


def auction_ajax_url(auction_id: int, type_: str, page: int = 1) -> str:
    return (
        "https://www.tibia.com/websiteservices/handle_charactertrades.php"
        f"?auctionid={auction_id}&type={type_}&currentpage={page}"
    )
