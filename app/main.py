"""FastAPI dashboard for Tibia Char Bazaar fair prices."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.predict import run_inference
from scrape.http import RateLimitedSession

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

app = FastAPI(title="Tibia Char Bazaar Fair Price", version="0.1.0")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
static_dir = Path(__file__).parent / "static"
# Vercel’s function FS is read-only — never mkdir there.
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
elif not __import__("os").environ.get("VERCEL"):
    static_dir.mkdir(exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_df: pd.DataFrame | None = None
_df_mtime: float | None = None
_metrics: dict[str, Any] | None = None
_metrics_mtime: float | None = None
_session = RateLimitedSession()


def get_df() -> pd.DataFrame:
    global _df, _df_mtime
    csv_path = DATA / "auctions.csv"
    if not csv_path.exists():
        raise FileNotFoundError("data/auctions.csv missing — run the pipeline first")
    mtime = csv_path.stat().st_mtime
    if _df is None or _df_mtime != mtime:
        _df = pd.read_csv(csv_path)
        _df_mtime = mtime
    return _df



def get_metrics() -> dict[str, Any]:
    global _metrics, _metrics_mtime
    path = MODELS / "metrics.json"
    if not path.exists():
        return {}
    mtime = path.stat().st_mtime
    if _metrics is None or _metrics_mtime != mtime:
        _metrics = json.loads(path.read_text())
        _metrics_mtime = mtime
    return _metrics or {}


def load_metrics_history() -> list[dict[str, Any]]:
    path = DATA / "metrics_history.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def count_detail_ok() -> int:
    path = DATA / "detail_auctions.jsonl"
    if not path.exists():
        return 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                if json.loads(line).get("detail_ok"):
                    n += 1
            except json.JSONDecodeError:
                continue
    return n


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"metrics": get_metrics()},
    )



@app.get("/api/overview")
def api_overview():
    import math
    import numpy as np

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    df = get_df()
    if "trainable_enriched" in df.columns and df["trainable_enriched"].any():
        train = df[df["trainable_enriched"] == True].copy()  # noqa: E712
    elif "trainable" in df.columns:
        train = df[df["trainable"] == True].copy()  # noqa: E712
    else:
        train = df.copy()

    by_voc = []
    if "vocation_base" in train.columns and "winning_bid" in train.columns:
        g = (
            train.dropna(subset=["winning_bid"])
            .groupby("vocation_base", dropna=False)["winning_bid"]
            .agg(["count", "median", "mean"])
            .reset_index()
            .rename(columns={"vocation_base": "vocation"})
        )
        by_voc = g.replace({np.nan: None}).to_dict(orient="records")

    by_level = []
    if "level" in train.columns and "winning_bid" in train.columns:
        tmp = train.dropna(subset=["level", "winning_bid"]).copy()
        if not tmp.empty:
            tmp["level_bin"] = pd.cut(
                tmp["level"],
                bins=[0, 100, 200, 300, 400, 500, 600, 800, 1000, 5000],
                right=False,
            )
            bl = (
                tmp.groupby("level_bin", observed=False)["winning_bid"]
                .agg(["count", "median"])
                .reset_index()
            )
            bl["level_bin"] = bl["level_bin"].astype(str)
            by_level = bl.replace({np.nan: None}).to_dict(orient="records")

    imp_path = MODELS / "feature_importance.csv"
    importance = []
    if imp_path.exists():
        importance = pd.read_csv(imp_path).head(20).replace({np.nan: None}).to_dict(orient="records")

    prices = []
    if "winning_bid" in train.columns and len(train):
        s = train["winning_bid"].dropna().astype(float)
        if len(s):
            hi = float(s.quantile(0.99))
            prices = s.clip(upper=hi).tolist()

    payload = {
        "n_rows": int(len(df)),
        "n_trainable": int(train.shape[0]),
        "by_vocation": by_voc,
        "by_level": by_level,
        "feature_importance": importance,
        "metrics": get_metrics(),
        "price_sample": prices,
    }
    return clean(payload)


@app.get("/api/auctions")
def api_auctions(
    vocation: str | None = None,
    world: str | None = None,
    min_level: int | None = None,
    max_level: int | None = None,
    min_bid: float | None = None,
    max_bid: float | None = None,
    q: str | None = None,
    trainable_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    df = get_df()
    view = df.copy()
    if trainable_only and "trainable" in view.columns:
        view = view[view["trainable"] == True]  # noqa: E712
    if vocation:
        if "vocation_base" in view.columns:
            view = view[view["vocation_base"].str.lower() == vocation.lower()]
        else:
            view = view[view["vocation"].astype(str).str.contains(vocation, case=False, na=False)]
    if world:
        view = view[view["world"].astype(str).str.contains(world, case=False, na=False)]
    if min_level is not None:
        view = view[view["level"] >= min_level]
    if max_level is not None:
        view = view[view["level"] <= max_level]
    if min_bid is not None and "winning_bid" in view.columns:
        view = view[view["winning_bid"] >= min_bid]
    if max_bid is not None and "winning_bid" in view.columns:
        view = view[view["winning_bid"] <= max_bid]
    if q:
        mask = view["name"].astype(str).str.contains(q, case=False, na=False)
        if "sales_arguments_text" in view.columns:
            mask = mask | view["sales_arguments_text"].astype(str).str.contains(q, case=False, na=False)
        view = view[mask]
    total = int(len(view))
    cols = [
        c
        for c in [
            "auction_id",
            "name",
            "level",
            "vocation",
            "world",
            "winning_bid",
            "bid",
            "status",
            "bid_type",
            "auction_end",
            "skill_magic_level",
            "max_combat_skill",
            "charm_points_total",
        ]
        if c in view.columns
    ]
    sort_col = "winning_bid" if "winning_bid" in view.columns else "bid"
    view = view.sort_values(sort_col, ascending=False, na_position="last")
    rows = view.iloc[offset : offset + limit][cols].fillna("").to_dict(orient="records")
    return {"total": total, "rows": rows}


@app.post("/api/predict")
async def api_predict(payload: dict[str, Any]):
    ref = (payload or {}).get("auction") or (payload or {}).get("url") or ""
    if not ref:
        raise HTTPException(400, "Provide auction URL or auction id")
    try:
        result = run_inference(str(ref), get_df(), MODELS, session=_session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse(result)


@app.get("/api/metrics")
def api_metrics():
    return get_metrics()


@app.get("/api/metrics-history")
def api_metrics_history():
    rows = load_metrics_history()
    n_enr = count_detail_ok()
    last_cp = 0
    for row in rows:
        label = str(row.get("checkpoint_label") or row.get("note") or "")
        n = int(row.get("n_enriched_sold") or 0)
        floor = (n // 1000) * 1000
        if label.startswith("enriched="):
            try:
                floor = max(floor, int(label.split("=", 1)[1]))
            except ValueError:
                pass
        if not label.startswith("seed"):
            last_cp = max(last_cp, floor)
        else:
            last_cp = max(last_cp, floor)
    return {
        "history": rows,
        "n_enriched_sold": n_enr,
        "last_checkpoint": last_cp,
        "next_retrain_at": last_cp + 1000,
    }


# ASGI entry: `uvicorn app.main:app`
