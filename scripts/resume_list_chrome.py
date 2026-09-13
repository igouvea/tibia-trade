#!/usr/bin/env python3
"""Resume past-trade list scrape via Chrome throttle (append-only, skip seen IDs)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrape.chrome_fetch import ChromeBlocked, ChromeSession
from scrape.http import auction_list_url
from scrape.list_scraper import _serialize_entry
from tibiapy.parsers import CharacterBazaarParser

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("resume_list_chrome")


def load_seen_ids(path: Path) -> set[int]:
    seen: set[int] = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                aid = json.loads(line).get("auction_id")
            except json.JSONDecodeError:
                continue
            if aid is not None:
                seen.add(int(aid))
    return seen


def estimate_start_page(n_rows: int, page_size: int = 25) -> int:
    # pages 1..N fully written ≈ n_rows / 25; resume at next page
    return max(1, (n_rows // page_size) + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-page", type=int, default=None)
    ap.add_argument("--max-pages", type=int, default=None, help="Cap pages this run")
    ap.add_argument("--min-delay", type=float, default=4.0)
    ap.add_argument("--max-delay", type=float, default=6.5)
    ap.add_argument("--target-pages", type=int, default=913)
    args = ap.parse_args()

    data = ROOT / "data"
    out = data / "list_auctions.jsonl"
    seen = load_seen_ids(out)
    start = args.start_page or estimate_start_page(len(seen))
    session = ChromeSession(min_delay=args.min_delay, max_delay=args.max_delay)

    log.info(
        "Chrome list resume: start_page=%s seen_ids=%s target_pages=%s delay=%.1f-%.1fs",
        start,
        len(seen),
        args.target_pages,
        args.min_delay,
        args.max_delay,
    )

    page = start
    pages_done = 0
    new_rows = 0
    backoff = args.min_delay
    total_pages = args.target_pages

    while True:
        if args.max_pages is not None and pages_done >= args.max_pages:
            log.info("Hit --max-pages=%s", args.max_pages)
            break
        if total_pages and page > total_pages:
            log.info("Reached target_pages=%s", total_pages)
            break

        url = auction_list_url(page)
        try:
            html = session.get(url)
            if "429" in html[:800] or "Too Many Requests" in html[:800]:
                raise ChromeBlocked("429 in body")
            bazaar = CharacterBazaarParser.from_content(html)
            backoff = args.min_delay
        except ChromeBlocked as exc:
            backoff = min(backoff * 2, 120)
            log.warning("Blocked page=%s (%s) — sleep %.1fs", page, exc, backoff)
            time.sleep(backoff)
            continue
        except Exception as exc:  # noqa: BLE001
            backoff = min(backoff * 2, 90)
            log.warning("Fail page=%s: %s — sleep %.1fs", page, exc, backoff)
            time.sleep(backoff)
            continue

        total_pages = bazaar.total_pages or total_pages
        if not bazaar.entries:
            log.warning("No entries on page %s — stopping", page)
            break

        added = 0
        with out.open("a", encoding="utf-8") as f:
            for entry in bazaar.entries:
                row = _serialize_entry(entry)
                aid = row.get("auction_id")
                if aid is None or int(aid) in seen:
                    continue
                seen.add(int(aid))
                f.write(json.dumps(row, default=str) + "\n")
                added += 1
                new_rows += 1

        pages_done += 1
        log.info(
            "Page %s/%s — +%s new (cum unique≈%s, new_this_run=%s)",
            page,
            total_pages,
            added,
            len(seen),
            new_rows,
        )
        page += 1

    log.info("DONE list resume pages_done=%s new_rows=%s unique≈%s", pages_done, new_rows, len(seen))


if __name__ == "__main__":
    main()
