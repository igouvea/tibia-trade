#!/usr/bin/env python3
"""Train models from current list+detail JSONL (enriched sold only)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.build import build_dataframe, prepare_xy
from model.train import train_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def main() -> None:
    data = ROOT / "data"
    df = build_dataframe(data / "list_auctions.jsonl", data / "detail_auctions.jsonl")
    df.to_csv(data / "auctions.csv", index=False)
    try:
        df.to_parquet(data / "auctions.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet: %s", exc)
    n_enr = int(df["enriched"].sum()) if "enriched" in df.columns else 0
    n_te = int(df["trainable_enriched"].sum()) if "trainable_enriched" in df.columns else 0
    log.info("rows=%s enriched=%s sold_enriched=%s", len(df), n_enr, n_te)
    if n_te < 50:
        raise SystemExit(f"Need >=50 sold+enriched to train (have {n_te})")
    X, y, meta = prepare_xy(df, require_enriched=True)
    meta.to_csv(data / "train_meta.csv", index=False)
    metrics = train_models(X, y, ROOT / "models", meta=meta)
    log.info("metrics %s", metrics)


if __name__ == "__main__":
    main()
