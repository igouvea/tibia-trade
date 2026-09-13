#!/usr/bin/env python3
"""Watch detail enrichment progress; retrain every 1000 successful sold details."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("retrain_watcher")

DATA = ROOT / "data"
MODELS = ROOT / "models"
DETAIL = DATA / "detail_auctions.jsonl"
LIST = DATA / "list_auctions.jsonl"
HISTORY = DATA / "metrics_history.jsonl"
METRICS = MODELS / "metrics.json"
CHECKPOINTS = MODELS / "checkpoints"
STATE = DATA / "retrain_watcher_state.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_detail_ok(path: Path = DETAIL) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("detail_ok"):
                n += 1
    return n


def load_history(path: Path = HISTORY) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def append_history(row: dict, path: Path = HISTORY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def last_checkpoint(history: list[dict]) -> int:
    """Highest completed 1000-floor checkpoint (seed entries with floor 0 ignored)."""
    best = 0
    for row in history:
        label = str(row.get("checkpoint_label") or row.get("note") or "")
        n = int(row.get("n_enriched_sold") or 0)
        floor = (n // 1000) * 1000
        if label.startswith("enriched="):
            try:
                floor = max(floor, int(label.split("=", 1)[1]))
            except ValueError:
                pass
        if label.startswith("seed"):
            # seed does not consume a 1000-checkpoint slot unless already >=1000
            floor = (n // 1000) * 1000
        best = max(best, floor)
    return best


def seed_history_if_empty() -> None:
    if HISTORY.exists() and HISTORY.stat().st_size > 0:
        return
    if not METRICS.exists():
        log.warning("No metrics.json to seed from; history stays empty until first retrain")
        return
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    n_enr = count_detail_ok()
    top = (metrics.get("top_features") or [])[:10]
    champ = metrics.get("champion") or "lightgbm"
    champ_m = (metrics.get("champion_metrics") or metrics.get("models", {}).get(champ) or metrics.get("lightgbm") or {})
    row = {
        "timestamp": utc_now(),
        "n_enriched_sold": n_enr,
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
        "champion": champ,
        "leaderboard": metrics.get("leaderboard") or [],
        "champion_metrics": {
            "MAE": champ_m.get("MAE"),
            "MAPE": champ_m.get("MAPE"),
            "R2_price": champ_m.get("R2_price"),
            "MedAE": champ_m.get("MedAE"),
        },
        "lightgbm": {
            "MAE": (metrics.get("lightgbm") or {}).get("MAE"),
            "MAPE": (metrics.get("lightgbm") or {}).get("MAPE"),
            "R2_price": (metrics.get("lightgbm") or {}).get("R2_price"),
        },
        "ridge": {
            "MAE": (metrics.get("ridge") or {}).get("MAE"),
            "MAPE": (metrics.get("ridge") or {}).get("MAPE"),
            "R2_price": (metrics.get("ridge") or {}).get("R2_price"),
        },
        "top_features": top,
        "note": "seed_from_metrics",
        "checkpoint_label": "seed_from_metrics",
    }
    append_history(row)
    log.info(
        "Seeded metrics_history.jsonl from models/metrics.json (n_enriched_sold=%s)",
        n_enr,
    )


def run_retrain(checkpoint_n: int, n_enriched: int) -> dict:
    from features.build import build_dataframe, prepare_xy
    from model.train import train_models

    log.info("=== RETRAIN checkpoint enriched=%s (detail_ok=%s) ===", checkpoint_n, n_enriched)
    df = build_dataframe(LIST, DETAIL)
    df.to_csv(DATA / "auctions.csv", index=False)
    try:
        df.to_parquet(DATA / "auctions.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("parquet: %s", exc)

    n_te = int(df["trainable_enriched"].sum()) if "trainable_enriched" in df.columns else 0
    if n_te < 50:
        raise RuntimeError(f"Need >=50 sold+enriched to train (have {n_te})")

    X, y, meta = prepare_xy(df, require_enriched=True)
    meta.to_csv(DATA / "train_meta.csv", index=False)
    metrics = train_models(X, y, MODELS, meta=meta)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    snap = {
        "timestamp": utc_now(),
        "checkpoint_label": f"enriched={checkpoint_n}",
        "n_enriched_sold": n_enriched,
        **metrics,
    }
    snap_path = CHECKPOINTS / f"metrics_at_{checkpoint_n}.json"
    snap_path.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")

    bundle_src = MODELS / "model_bundle.joblib"
    if bundle_src.exists():
        bundle_dir = CHECKPOINTS / "bundles"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bundle_src, bundle_dir / f"model_bundle_at_{checkpoint_n}.joblib")

    champ = metrics.get("champion") or "lightgbm"
    champ_m = (metrics.get("champion_metrics") or metrics.get("models", {}).get(champ) or metrics.get("lightgbm") or {})
    hist = {
        "timestamp": utc_now(),
        "n_enriched_sold": n_enriched,
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
        "champion": champ,
        "leaderboard": metrics.get("leaderboard") or [],
        "champion_metrics": {
            "MAE": champ_m.get("MAE"),
            "MedAE": champ_m.get("MedAE"),
            "MAPE": champ_m.get("MAPE"),
            "MdAPE": champ_m.get("MdAPE"),
            "R2_price": champ_m.get("R2_price"),
        },
        "lightgbm": {
            "MAE": (metrics.get("lightgbm") or {}).get("MAE"),
            "MedAE": (metrics.get("lightgbm") or {}).get("MedAE"),
            "MAPE": (metrics.get("lightgbm") or {}).get("MAPE"),
            "MdAPE": (metrics.get("lightgbm") or {}).get("MdAPE"),
            "R2_price": (metrics.get("lightgbm") or {}).get("R2_price"),
            "slices": (metrics.get("lightgbm") or {}).get("slices"),
        },
        "ridge": {
            "MAE": (metrics.get("ridge") or {}).get("MAE"),
            "MedAE": (metrics.get("ridge") or {}).get("MedAE"),
            "MAPE": (metrics.get("ridge") or {}).get("MAPE"),
            "MdAPE": (metrics.get("ridge") or {}).get("MdAPE"),
            "R2_price": (metrics.get("ridge") or {}).get("R2_price"),
            "slices": (metrics.get("ridge") or {}).get("slices"),
        },
        "top_features": (metrics.get("top_features") or [])[:10],
        "note": f"retrain at enriched={checkpoint_n}",
        "checkpoint_label": f"enriched={checkpoint_n}",
        "n_trainable_enriched": n_te,
    }
    # Also copy living models into checkpoint
    living_src = MODELS / "living"
    if living_src.exists():
        living_dst = CHECKPOINTS / "living" / f"at_{checkpoint_n}"
        if living_dst.exists():
            shutil.rmtree(living_dst)
        shutil.copytree(living_src, living_dst)
    append_history(hist)
    log.info(
        "Retrain done enriched=%s n_train=%s champion=%s MAE=%.1f MAPE=%.1f → %s + history",
        checkpoint_n,
        metrics.get("n_train"),
        champ,
        champ_m.get("MAE") or 0,
        champ_m.get("MAPE") or 0,
        snap_path,
    )
    return hist


def save_state(count: int, last_cp: int) -> None:
    STATE.write_text(
        json.dumps(
            {
                "updated_at": utc_now(),
                "n_enriched_sold": count,
                "last_checkpoint": last_cp,
                "next_retrain_at": last_cp + 1000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def tick() -> None:
    seed_history_if_empty()
    history = load_history()
    count = count_detail_ok()
    last_cp = last_checkpoint(history)
    floor = (count // 1000) * 1000
    next_at = last_cp + 1000
    save_state(count, last_cp)
    log.info(
        "watch: detail_ok=%s last_checkpoint=%s next_retrain_at=%s",
        count,
        last_cp,
        next_at,
    )
    if floor >= 1000 and floor > last_cp:
        # Catch up one checkpoint at a time (current data for each missed floor)
        for cp in range(last_cp + 1000, floor + 1, 1000):
            run_retrain(cp, count)
            last_cp = cp
            save_state(count, last_cp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30.0, help="Poll seconds")
    ap.add_argument("--once", action="store_true", help="Single tick then exit")
    args = ap.parse_args()

    log.info(
        "Retrain watcher started interval=%ss detail=%s history=%s",
        args.interval,
        DETAIL,
        HISTORY,
    )
    seed_history_if_empty()

    while True:
        try:
            tick()
        except Exception as exc:  # noqa: BLE001
            log.exception("tick failed: %s", exc)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
