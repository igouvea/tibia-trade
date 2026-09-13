#!/usr/bin/env python3
"""Tiny HTML fetch proxy for Vercel appraisals (runs where Cloudflare allows Tibia)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse

from scrape.http import DEFAULT_UA, _looks_blocked, auction_detail_urls

app = FastAPI(title="tibia-trade fetch proxy")
TOKEN = os.environ.get("TIBIA_FETCH_PROXY_TOKEN") or ""


def _auth(authorization: str | None) -> None:
    if not TOKEN:
        return
    if not authorization or authorization.removeprefix("Bearer ").strip() != TOKEN:
        raise HTTPException(401, "unauthorized")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/fetch")
def fetch(
    url: str | None = None,
    auction_id: int | None = None,
    authorization: str | None = Header(default=None),
):
    _auth(authorization)
    urls: list[str] = []
    if url:
        urls.append(unquote(url))
    if auction_id:
        urls.extend(auction_detail_urls(int(auction_id)))
    if not urls:
        raise HTTPException(400, "url or auction_id required")

    # Prefer curl_cffi then cloudscraper (same as main fetch)
    errors = []
    try:
        from curl_cffi import requests as crequests

        for u in urls:
            try:
                r = crequests.get(u, impersonate="chrome131", timeout=25, headers={"User-Agent": DEFAULT_UA})
                if r.status_code == 200 and r.text and not _looks_blocked(r.text, r.status_code):
                    return PlainTextResponse(r.text, media_type="text/html")
                errors.append(f"curl_cffi {r.status_code}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"curl_cffi {exc}")
    except ImportError:
        errors.append("curl_cffi missing")

    import cloudscraper

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )
    for u in urls:
        try:
            r = scraper.get(u, timeout=25)
            if r.status_code == 200 and r.text and not _looks_blocked(r.text, r.status_code):
                return PlainTextResponse(r.text, media_type="text/html")
            errors.append(f"cloudscraper {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"cloudscraper {exc}")

    raise HTTPException(502, "; ".join(errors[:4]))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8787"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
