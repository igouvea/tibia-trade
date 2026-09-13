"""Enrich auctions with detail-page skills/charms/highlights."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from scrape.detail_parser import parse_detail_html
from scrape.http import RateLimitedSession, auction_detail_url

log = logging.getLogger(__name__)


def scrape_details(
    auction_ids: Iterable[int],
    session: RateLimitedSession | None = None,
    out_jsonl: Path | None = None,
    skip_ids: set[int] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    session = session or RateLimitedSession(min_delay=0.55, max_delay=1.2)
    skip_ids = skip_ids or set()
    rows: list[dict[str, Any]] = []
    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    done = 0
    for aid in auction_ids:
        if aid in skip_ids:
            continue
        if limit is not None and done >= limit:
            break
        url = auction_detail_url(aid)
        try:
            html = session.get(url)
            row = parse_detail_html(html, auction_id=aid)
            row["detail_ok"] = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Detail failed for %s: %s", aid, exc)
            row = {"auction_id": aid, "detail_ok": False, "error": str(exc)}
        rows.append(row)
        done += 1
        if out_jsonl:
            with out_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        if done % 25 == 0:
            log.info("Details scraped: %s", done)
    return rows


def load_detail_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(int(json.loads(line).get("auction_id")))
            except Exception:  # noqa: BLE001
                continue
    return ids
