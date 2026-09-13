#!/usr/bin/env python3
"""Chrome-only detail enrichment with resume + backoff. Prefer sold (Winning Bid).

Ordering: stratified-by-bid-quintile round-robin (NOT richest-first).
Still enriches Winning Bid + finished only until that pool is exhausted.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scrape.chrome_fetch import ChromeBlocked, chrome_dump
from scrape.detail_parser import parse_detail_html
from scrape.detail_scraper import load_detail_ids
from scrape.http import auction_detail_url

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("enrich_chrome")


def load_sold_ids(list_jsonl: Path, seed: int = 42) -> list[int]:
    """Return Winning Bid + finished auction ids in stratified bid-quintile order.

    Still sold-only (user rule). Previously sorted by bid DESC which caused the
    first ~500 enriched rows to be exclusively top-quintile riches and skewed
    the fair-price model. Now: assign each sold row to a bid quintile, shuffle
    within each quintile, then round-robin across quintiles so mid/low prices
    enter the enriched pool alongside high ones.
    """
    rows = []
    with list_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            status = str(r.get("status") or "").upper()
            bid_type = str(r.get("bid_type") or "").upper()
            if "CANCEL" in status:
                continue
            if "FINISHED" in status and "WINNING" in bid_type:
                rows.append(r)
    if not rows:
        return []
    bids = sorted(float(r.get("bid") or 0) for r in rows)
    # Quintile edges from all sold bids
    def q_edge(p: float) -> float:
        idx = min(len(bids) - 1, max(0, int(round((len(bids) - 1) * p))))
        return bids[idx]
    edges = [q_edge(p) for p in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    buckets: list[list[dict]] = [[] for _ in range(5)]
    for r in rows:
        b = float(r.get("bid") or 0)
        placed = False
        for i in range(5):
            lo, hi = edges[i], edges[i + 1]
            if i < 4:
                if lo <= b < hi or (i == 0 and b <= lo):
                    buckets[i].append(r)
                    placed = True
                    break
            else:
                if b >= lo:
                    buckets[i].append(r)
                    placed = True
                    break
        if not placed:
            buckets[-1].append(r)
    rng = random.Random(seed)
    for bucket in buckets:
        rng.shuffle(bucket)
    # Round-robin across quintiles
    ordered: list[dict] = []
    pointers = [0] * 5
    while True:
        progressed = False
        for i in range(5):
            if pointers[i] < len(buckets[i]):
                ordered.append(buckets[i][pointers[i]])
                pointers[i] += 1
                progressed = True
        if not progressed:
            break
    log.info(
        "Sold pool order: stratified bid-quintile round-robin (seed=%s); "
        "bucket sizes=%s edges=%s",
        seed,
        [len(b) for b in buckets],
        [round(e, 1) for e in edges],
    )
    return [int(r["auction_id"]) for r in ordered]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-delay", type=float, default=4.0)
    ap.add_argument("--max-delay", type=float, default=6.5)
    ap.add_argument("--rebuild-csv-every", type=int, default=50)
    args = ap.parse_args()

    data = ROOT / "data"
    list_jsonl = data / "list_auctions.jsonl"
    detail_jsonl = data / "detail_auctions.jsonl"
    detail_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Drop failed stubs so we retry them
    if detail_jsonl.exists():
        kept = []
        for line in detail_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("detail_ok"):
                kept.append(json.dumps(d, default=str))
        detail_jsonl.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    already = load_detail_ids(detail_jsonl)
    sold = load_sold_ids(list_jsonl)
    pending = [i for i in sold if i not in already]
    if args.limit is not None:
        pending = pending[: args.limit]

    log.info(
        "Chrome enrich: sold=%s already_ok=%s pending=%s delay=%.1f-%.1fs",
        len(sold),
        len(already),
        len(pending),
        args.min_delay,
        args.max_delay,
    )

    ok = fail = 0
    backoff = args.min_delay
    last = 0.0

    for idx, aid in enumerate(pending, 1):
        # throttle
        elapsed = time.monotonic() - last
        wait = random.uniform(args.min_delay, args.max_delay) - elapsed
        # stretch if we recently backed off
        wait = max(wait, 0) + max(0, backoff - args.min_delay) * 0.25
        if wait > 0:
            time.sleep(wait)

        url = auction_detail_url(aid)
        try:
            html = chrome_dump(url)
            if "429" in html[:500] or "Too Many Requests" in html[:500]:
                raise ChromeBlocked("429 in body")
            row = parse_detail_html(html, auction_id=aid)
            if not row.get("skills") and not row.get("level"):
                raise RuntimeError("empty parse")
            row["detail_ok"] = True
            ok += 1
            backoff = args.min_delay
        except ChromeBlocked as exc:
            fail += 1
            backoff = min(backoff * 2, 120)
            log.warning("Blocked aid=%s (%s) — backoff %.1fs", aid, exc, backoff)
            row = {"auction_id": aid, "detail_ok": False, "error": str(exc)}
            time.sleep(backoff)
            # do not persist failures — allow resume retry
            last = time.monotonic()
            continue
        except Exception as exc:  # noqa: BLE001
            fail += 1
            log.warning("Fail aid=%s: %s", aid, exc)
            last = time.monotonic()
            # skip persist of failures
            continue

        with detail_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
        last = time.monotonic()

        if idx % 10 == 0 or idx == 1:
            rate = ok / max(ok + fail, 1)
            log.info(
                "progress %s/%s ok=%s fail=%s success_rate=%.0f%% last=%s items=%s store=%s",
                idx,
                len(pending),
                ok,
                fail,
                100 * rate,
                row.get("name"),
                row.get("items_count"),
                row.get("store_items_count"),
            )

        if args.rebuild_csv_every and ok % args.rebuild_csv_every == 0:
            try:
                from features.build import build_dataframe

                df = build_dataframe(list_jsonl, detail_jsonl)
                df.to_csv(data / "auctions.csv", index=False)
                log.info(
                    "CSV refresh rows=%s enriched=%s",
                    len(df),
                    int(df["enriched"].sum()) if "enriched" in df.columns else "?",
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("CSV rebuild failed: %s", exc)

    # final csv
    try:
        from features.build import build_dataframe

        df = build_dataframe(list_jsonl, detail_jsonl)
        df.to_csv(data / "auctions.csv", index=False)
        log.info(
            "DONE ok=%s fail=%s enriched=%s → %s",
            ok,
            fail,
            int(df["enriched"].sum()) if "enriched" in df.columns else 0,
            data / "auctions.csv",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("final CSV failed: %s", exc)

    log.info("success_rate=%.1f%%", 100 * ok / max(ok + fail, 1))


if __name__ == "__main__":
    main()
