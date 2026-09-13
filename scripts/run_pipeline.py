#!/usr/bin/env python3
"""End-to-end: scrape lists → enrich details → build dataset → train."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.build import build_dataframe, prepare_xy
from model.train import train_models
from scrape.detail_scraper import load_detail_ids, scrape_details
from scrape.http import RateLimitedSession
from scrape.list_scraper import scrape_list_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("pipeline")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-list", action="store_true")
    p.add_argument("--skip-details", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--fresh-list", action="store_true")
    p.add_argument("--max-list-pages", type=int, default=None)
    p.add_argument("--detail-limit", type=int, default=None)
    p.add_argument(
        "--detail-priority",
        choices=["all_finished", "winning_only"],
        default="winning_only",
    )
    args = p.parse_args()

    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    list_jsonl = data / "list_auctions.jsonl"
    detail_jsonl = data / "detail_auctions.jsonl"
    csv_path = data / "auctions.csv"
    parquet_path = data / "auctions.parquet"
    models_dir = ROOT / "models"

    session = RateLimitedSession()

    if not args.skip_list:
        if args.fresh_list and list_jsonl.exists():
            list_jsonl.unlink()
        scrape_list_pages(
            session=session,
            max_pages=args.max_list_pages,
            out_jsonl=list_jsonl,
        )
    else:
        log.info("Skipping list scrape; using %s", list_jsonl)

    # Build interim frame to choose detail IDs
    df = build_dataframe(list_jsonl, detail_jsonl if detail_jsonl.exists() else None)

    if not args.skip_details:
        if args.detail_priority == "winning_only":
            ids = (
                df.loc[df["trainable"], "auction_id"]
                .astype(int)
                .tolist()
            )
        else:
            ids = df.loc[df["is_finished"], "auction_id"].astype(int).tolist()
        # Prefer higher bids first (more informative)
        ids = (
            df.loc[df["auction_id"].isin(ids)]
            .assign(_bid=lambda x: x["bid"].fillna(0))
            .sort_values("_bid", ascending=False)["auction_id"]
            .astype(int)
            .tolist()
        )
        already = load_detail_ids(detail_jsonl)
        log.info("Detail queue: %s ids (%s already done)", len(ids), len(already))
        scrape_details(
            ids,
            session=session,
            out_jsonl=detail_jsonl,
            skip_ids=already,
            limit=args.detail_limit,
        )
        df = build_dataframe(list_jsonl, detail_jsonl)
    else:
        log.info("Skipping details")

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Parquet write failed: %s", exc)

    n_total = len(df)
    n_train = int(df["trainable"].sum()) if "trainable" in df.columns else 0
    n_detail = int(df["detail_ok"].fillna(False).sum()) if "detail_ok" in df.columns else 0
    log.info("Dataset rows=%s trainable=%s with_details=%s → %s", n_total, n_train, n_detail, csv_path)

    if not args.skip_train:
        X, y, meta = prepare_xy(df, require_enriched=True)
        meta.to_csv(data / "train_meta.csv", index=False)
        metrics = train_models(X, y, models_dir)
        log.info("Metrics: %s", metrics)
    else:
        log.info("Skipping train")


if __name__ == "__main__":
    main()
