#!/usr/bin/env python3
"""Enrich auctions with full detail pages (resume-safe). Prioritize sold (Winning Bid)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.build import build_dataframe
from scrape.detail_scraper import load_detail_ids, scrape_details
from scrape.http import RateLimitedSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("enrich")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="Max new details to fetch")
    ap.add_argument(
        "--priority",
        choices=["sold", "finished", "all"],
        default="sold",
        help="sold=finished+Winning Bid first (default)",
    )
    ap.add_argument("--min-delay", type=float, default=0.55)
    ap.add_argument("--max-delay", type=float, default=1.25)
    args = ap.parse_args()

    data = ROOT / "data"
    list_jsonl = data / "list_auctions.jsonl"
    detail_jsonl = data / "detail_auctions.jsonl"
    if not list_jsonl.exists():
        raise SystemExit(f"Missing {list_jsonl}")

    df = build_dataframe(list_jsonl, detail_jsonl if detail_jsonl.exists() else None)
    if args.priority == "sold":
        pool = df[df["sold"]]
    elif args.priority == "finished":
        pool = df[df["is_finished"] & ~df["is_cancelled"]]
    else:
        pool = df

    # Highest bid first among sold; else by level
    sort_col = "winning_bid" if "winning_bid" in pool.columns else "bid"
    ids = (
        pool.assign(_s=pool[sort_col].fillna(pool.get("level", 0)))
        .sort_values("_s", ascending=False)["auction_id"]
        .astype(int)
        .tolist()
    )
    already = load_detail_ids(detail_jsonl)
    # Drop details that failed previously so we can retry? keep skip of all present
    pending = [i for i in ids if i not in already]
    log.info(
        "Enrich priority=%s pool=%s already=%s pending=%s limit=%s",
        args.priority,
        len(ids),
        len(already),
        len(pending),
        args.limit,
    )
    session = RateLimitedSession(min_delay=args.min_delay, max_delay=args.max_delay)
    scrape_details(
        pending,
        session=session,
        out_jsonl=detail_jsonl,
        skip_ids=set(),
        limit=args.limit,
    )
    # Rebuild CSV snapshot
    df2 = build_dataframe(list_jsonl, detail_jsonl)
    csv_path = data / "auctions.csv"
    df2.to_csv(csv_path, index=False)
    n_enr = int(df2["enriched"].sum()) if "enriched" in df2.columns else 0
    n_sold_enr = int(df2["trainable_enriched"].sum()) if "trainable_enriched" in df2.columns else 0
    log.info("Wrote %s rows=%s enriched=%s sold_enriched=%s", csv_path, len(df2), n_enr, n_sold_enr)


if __name__ == "__main__":
    main()
