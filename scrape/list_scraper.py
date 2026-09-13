"""Scrape past character trade list pages."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tibiapy.parsers import CharacterBazaarParser

from scrape.http import RateLimitedSession, auction_list_url

log = logging.getLogger(__name__)


def _serialize_entry(entry) -> dict[str, Any]:
    d = entry.model_dump()
    # Normalize enums / datetimes for JSON
    for k, v in list(d.items()):
        if hasattr(v, "value"):
            d[k] = v.value if not isinstance(v, datetime) else v
        if isinstance(v, datetime):
            d[k] = v.astimezone(timezone.utc).isoformat()
    # Flatten nested bits we care about
    args = []
    for a in d.get("sales_arguments") or []:
        if isinstance(a, dict):
            args.append(a.get("content") or "")
        else:
            args.append(str(a))
    d["sales_arguments_text"] = " | ".join(args)
    items = d.get("displayed_items") or []
    d["displayed_item_names"] = ", ".join(
        (i.get("name") if isinstance(i, dict) else getattr(i, "name", "")) or ""
        for i in items
    )
    outfit = d.get("outfit") or {}
    d["outfit_id"] = outfit.get("outfit_id") if isinstance(outfit, dict) else None
    d["status"] = str(d.get("status") or "")
    d["bid_type"] = str(d.get("bid_type") or "")
    d["vocation"] = str(d.get("vocation") or "")
    d["sex"] = str(d.get("sex") or "")
    return d


def scrape_list_pages(
    session: RateLimitedSession | None = None,
    start_page: int = 1,
    max_pages: int | None = None,
    out_jsonl: Path | None = None,
) -> list[dict[str, Any]]:
    session = session or RateLimitedSession()
    rows: list[dict[str, Any]] = []
    page = start_page
    total_pages = None
    seen_ids: set[int] = set()

    if out_jsonl:
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    while True:
        url = auction_list_url(page)
        log.info("Fetching list page %s ...", page)
        try:
            html = session.get(url)
            bazaar = CharacterBazaarParser.from_content(html)
        except Exception as exc:  # noqa: BLE001
            log.warning("List page %s failed (%s) — back off 25s and retry", page, exc)
            import time as _time
            _time.sleep(25)
            html = session.get(url)
            bazaar = CharacterBazaarParser.from_content(html)
        total_pages = bazaar.total_pages or total_pages
        if not bazaar.entries:
            log.warning("No entries on page %s — stopping", page)
            break
        for entry in bazaar.entries:
            row = _serialize_entry(entry)
            aid = row.get("auction_id")
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            rows.append(row)
            if out_jsonl:
                with out_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
        log.info(
            "Page %s/%s — %s entries (cum %s unique)",
            page,
            total_pages,
            len(bazaar.entries),
            len(rows),
        )
        if max_pages and page >= start_page + max_pages - 1:
            break
        if total_pages and page >= total_pages:
            break
        page += 1

    return rows


def iter_existing_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
