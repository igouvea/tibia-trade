"""Runtime inference with multi-model champion support (Vercel-friendly)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_bundle(models_dir: Path) -> dict[str, Any]:
    models_dir = Path(models_dir)
    vercel = models_dir / "model_bundle_vercel.joblib"
    full = models_dir / "model_bundle.joblib"
    if os.environ.get("VERCEL") and vercel.exists():
        return joblib.load(vercel)
    if full.exists():
        try:
            return joblib.load(full)
        except Exception:
            if vercel.exists():
                return joblib.load(vercel)
            raise
    if vercel.exists():
        return joblib.load(vercel)
    raise FileNotFoundError(f"No model bundle in {models_dir}")


def _resolve_model(bundle: dict[str, Any]) -> tuple[str, Any]:
    models = bundle.get("models") or {}
    champion = bundle.get("champion")
    if champion and champion in models and models[champion] is not None:
        return str(champion), models[champion]
    if bundle.get("lgbm") is not None:
        return "lightgbm", bundle["lgbm"]
    if bundle.get("ridge") is not None:
        return "ridge", bundle["ridge"]
    if models:
        name = next(iter(models))
        return str(name), models[name]
    raise RuntimeError("Bundle has no usable model")


def _drivers_for(model: Any, cols: list[str], x: pd.DataFrame) -> list[dict[str, Any]]:
    drivers: list[dict[str, Any]] = []
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        ranked = sorted(zip(cols, importances, x.iloc[0].tolist()), key=lambda t: -float(t[1]))
        for name, imp, val in ranked[:8]:
            if abs(float(val)) > 0 or float(imp) > 0:
                drivers.append({"feature": name, "value": float(val), "importance": float(imp)})
        return drivers
    pipe_model = None
    if hasattr(model, "named_steps"):
        pipe_model = model.named_steps.get("model")
    coefs = getattr(pipe_model or model, "coef_", None)
    if coefs is not None:
        ranked = sorted(zip(cols, coefs, x.iloc[0].tolist()), key=lambda t: -abs(float(t[1])))
        for name, coef, val in ranked[:8]:
            if abs(float(val)) > 0 or abs(float(coef)) > 0:
                drivers.append({"feature": name, "value": float(val), "importance": abs(float(coef))})
    return drivers


def predict_row(bundle: dict[str, Any], feature_row: dict[str, float]) -> dict[str, Any]:
    cols = bundle["feature_columns"]
    x = pd.DataFrame([{c: float(feature_row.get(c, 0.0) or 0.0) for c in cols}])
    model_name, model = _resolve_model(bundle)
    pred_log = float(model.predict(x)[0])
    zero = pd.DataFrame([{c: 0.0 for c in cols}])
    baseline = float(np.expm1(model.predict(zero)[0]))
    drivers = _drivers_for(model, cols, x)

    ridge = bundle.get("ridge")
    ridge_price = None
    if ridge is not None:
        try:
            ridge_price = float(np.expm1(ridge.predict(x)[0]))
        except Exception:
            ridge_price = None

    point = float(np.expm1(pred_log))
    low = point * 0.75
    high = point * 1.35
    GOLD_PER_TC = 41_000
    return {
        "fair_price": point,
        "fair_price_low": low,
        "fair_price_high": high,
        "ridge_price": ridge_price,
        "baseline_zero_features_price": baseline,
        "drivers": drivers,
        "pred_log": pred_log,
        "model_used": model_name,
        "champion": bundle.get("champion") or model_name,
        "leaderboard": bundle.get("leaderboard") or [],
        "gold_per_tc": GOLD_PER_TC,
        "fair_price_gold_equiv": point * GOLD_PER_TC,
        "fair_price_low_gold_equiv": low * GOLD_PER_TC,
        "fair_price_high_gold_equiv": high * GOLD_PER_TC,
    }
