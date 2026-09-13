"""Runtime inference without LightGBM (Vercel-friendly)."""
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
    # Prefer slim ridge bundle on Vercel / when lightgbm isn't installed.
    if os.environ.get("VERCEL") and vercel.exists():
        return joblib.load(vercel)
    if vercel.exists():
        try:
            import lightgbm  # noqa: F401
        except Exception:
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


def predict_row(bundle: dict[str, Any], feature_row: dict[str, float]) -> dict[str, Any]:
    cols = bundle["feature_columns"]
    x = pd.DataFrame([{c: float(feature_row.get(c, 0.0) or 0.0) for c in cols}])
    lgbm = bundle.get("lgbm")
    ridge = bundle["ridge"]
    ridge_log = float(ridge.predict(x)[0])

    if lgbm is not None:
        pred_log = float(lgbm.predict(x)[0])
        baseline = float(lgbm.predict(pd.DataFrame([{c: 0.0 for c in cols}]))[0])
        baseline = float(np.expm1(baseline))
        drivers = []
        importances = getattr(lgbm, "feature_importances_", None)
        if importances is not None:
            ranked = sorted(zip(cols, importances, x.iloc[0].tolist()), key=lambda t: -t[1])
            for name, imp, val in ranked[:8]:
                if abs(val) > 0 or imp > 0:
                    drivers.append({"feature": name, "value": val, "importance": int(imp)})
    else:
        pred_log = ridge_log
        baseline = float(np.expm1(ridge.predict(pd.DataFrame([{c: 0.0 for c in cols}]))[0]))
        drivers = []
        model = ridge.named_steps.get("model") if hasattr(ridge, "named_steps") else None
        coefs = getattr(model, "coef_", None)
        if coefs is not None:
            ranked = sorted(zip(cols, coefs, x.iloc[0].tolist()), key=lambda t: -abs(float(t[1])))
            for name, coef, val in ranked[:8]:
                if abs(float(val)) > 0 or abs(float(coef)) > 0:
                    drivers.append({"feature": name, "value": float(val), "importance": abs(float(coef))})

    point = float(np.expm1(pred_log))
    low = point * 0.75
    high = point * 1.35
    GOLD_PER_TC = 41_000
    return {
        "fair_price": point,
        "fair_price_low": low,
        "fair_price_high": high,
        "ridge_price": float(np.expm1(ridge_log)),
        "baseline_zero_features_price": baseline,
        "drivers": drivers,
        "pred_log": pred_log,
        "model_used": "lightgbm" if lgbm is not None else "ridge",
        "gold_per_tc": GOLD_PER_TC,
        "fair_price_gold_equiv": point * GOLD_PER_TC,
        "fair_price_low_gold_equiv": low * GOLD_PER_TC,
        "fair_price_high_gold_equiv": high * GOLD_PER_TC,
    }
