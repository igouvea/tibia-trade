"""Train baseline + gradient boosting on log1p(winning_bid).

Uses stratified train/test by log-bid bins, inverse-density sample weights
(so a rich-skewed enriched slice does not dominate), simpler LightGBM
hyperparameters for small n, and metrics broken down by bid quartile / vocation.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict[str, float]:
    yt = np.expm1(y_true_log)
    yp = np.expm1(y_pred_log)
    yp = np.clip(yp, 0, None)
    ae = np.abs(yt - yp)
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "MedAE": float(np.median(ae)),
        "MAPE": _mape(yt, yp),
        "MdAPE": _mdape(yt, yp),
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "R2_price": float(r2_score(yt, yp)),
        "n_test": int(len(yt)),
    }


def _log_bid_bins(y_log: pd.Series | np.ndarray, n_bins: int = 5) -> np.ndarray:
    y = np.asarray(y_log, dtype=float)
    # qcut can fail on ties — fall back to fewer bins
    for k in range(n_bins, 1, -1):
        try:
            cats = pd.qcut(y, q=k, labels=False, duplicates="drop")
            return np.asarray(cats, dtype=int)
        except ValueError:
            continue
    return np.zeros(len(y), dtype=int)


def _sample_weights(y_log: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Inverse frequency by log-bid bin — upweight scarce mid/low bids."""
    bins = _log_bid_bins(y_log, n_bins=n_bins)
    _, counts = np.unique(bins, return_counts=True)
    # map bin -> weight
    freq = pd.Series(bins).value_counts()
    w = np.array([1.0 / freq.loc[b] for b in bins], dtype=float)
    w *= len(w) / w.sum()
    return w


def _slice_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    meta: pd.DataFrame | None,
) -> dict[str, Any]:
    yt = np.expm1(y_true_log)
    yp = np.clip(np.expm1(y_pred_log), 0, None)
    out: dict[str, Any] = {"by_bid_quartile": [], "by_vocation": []}

    try:
        q = pd.qcut(yt, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    except ValueError:
        q = pd.Series(["all"] * len(yt))
    for label in pd.unique(q):
        mask = np.asarray(q == label)
        if not mask.any():
            continue
        m = _metrics(y_true_log[mask], y_pred_log[mask])
        m["quartile"] = str(label)
        m["bid_min"] = float(yt[mask].min())
        m["bid_max"] = float(yt[mask].max())
        m["bid_median"] = float(np.median(yt[mask]))
        out["by_bid_quartile"].append(m)

    if meta is not None and "vocation_base" in meta.columns:
        voc = meta["vocation_base"].astype(str).reset_index(drop=True)
    elif meta is not None and "vocation" in meta.columns:
        from features.build import VOCATION_BASE

        voc = meta["vocation"].map(
            lambda v: VOCATION_BASE.get(str(v), str(v).split()[-1] if v else "Unknown")
        ).reset_index(drop=True)
    else:
        voc = None
    if voc is not None:
        for vname, idx in voc.groupby(voc).groups.items():
            mask = np.zeros(len(yt), dtype=bool)
            mask[list(idx)] = True
            if mask.sum() < 3:
                continue
            m = _metrics(y_true_log[mask], y_pred_log[mask])
            m["vocation"] = str(vname)
            out["by_vocation"].append(m)
    return out


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_dir: Path,
    random_state: int = 42,
    meta: pd.DataFrame | None = None,
) -> dict[str, Any]:
    import lightgbm as lgb
    models_dir.mkdir(parents=True, exist_ok=True)
    n = len(X)
    bins = _log_bid_bins(y, n_bins=5)

    # Stratified split by log-bid bins (fallback to plain split)
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        train_idx, test_idx = next(splitter.split(X, bins))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        meta_test = meta.iloc[test_idx].reset_index(drop=True) if meta is not None else None
        split_mode = "stratified_log_bid_bins"
    except ValueError as exc:
        log.warning("Stratified split failed (%s) — using random split", exc)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
        meta_test = None
        split_mode = "random"

    w_train = _sample_weights(y_train.to_numpy(), n_bins=5)

    ridge = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=5.0)),
        ]
    )
    ridge.fit(X_train, y_train, model__sample_weight=w_train)
    ridge_pred = ridge.predict(X_test)
    ridge_metrics = _metrics(y_test.to_numpy(), ridge_pred)

    # Small-n LightGBM: shallow trees, high min_child_samples, fewer estimators
    min_child = max(25, n // 15)
    num_leaves = 15 if n < 800 else 31
    n_estimators = 120 if n < 800 else 250
    max_depth = 4 if n < 800 else 6

    lgbm = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=num_leaves,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=min_child,
        reg_lambda=1.0,
        random_state=random_state,
        verbosity=-1,
    )
    lgbm.fit(X_train, y_train, sample_weight=w_train)
    lgb_pred = lgbm.predict(X_test)
    lgb_metrics = _metrics(y_test.to_numpy(), lgb_pred)

    importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": lgbm.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(models_dir / "feature_importance.csv", index=False)

    coef = pd.DataFrame(
        {
            "feature": X.columns,
            "coefficient": ridge.named_steps["model"].coef_,
        }
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coef.to_csv(models_dir / "ridge_coefficients.csv", index=False)

    bundle = {
        "lgbm": lgbm,
        "ridge": ridge,
        "feature_columns": list(X.columns),
        "median_bid": float(np.expm1(y.median())),
        "train_config": {
            "split_mode": split_mode,
            "sample_weights": "inverse_log_bid_bin_freq",
            "lgbm": {
                "n_estimators": n_estimators,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "min_child_samples": min_child,
            },
        },
    }
    joblib.dump(bundle, models_dir / "model_bundle.joblib")

    lgb_slices = _slice_metrics(y_test.to_numpy(), lgb_pred, meta_test)
    ridge_slices = _slice_metrics(y_test.to_numpy(), ridge_pred, meta_test)

    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "n_total": int(n),
        "target": "log1p(winning_bid)",
        "split_mode": split_mode,
        "sample_weights": "inverse_log_bid_bin_freq",
        "lgbm_hyperparams": bundle["train_config"]["lgbm"],
        "lightgbm": {**lgb_metrics, **{"slices": lgb_slices}},
        "ridge": {**ridge_metrics, **{"slices": ridge_slices}},
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info(
        "LightGBM MAE=%.1f MedAE=%.1f MAPE=%.1f%% MdAPE=%.1f%% R2_price=%.3f",
        lgb_metrics["MAE"],
        lgb_metrics["MedAE"],
        lgb_metrics["MAPE"],
        lgb_metrics["MdAPE"],
        lgb_metrics["R2_price"],
    )
    log.info(
        "Ridge    MAE=%.1f MedAE=%.1f MAPE=%.1f%% MdAPE=%.1f%% R2_price=%.3f",
        ridge_metrics["MAE"],
        ridge_metrics["MedAE"],
        ridge_metrics["MAPE"],
        ridge_metrics["MdAPE"],
        ridge_metrics["R2_price"],
    )
    for row in lgb_slices.get("by_bid_quartile") or []:
        log.info(
            "  LGBM %s bids [%.0f–%.0f] MAE=%.0f MdAPE=%.1f%% n=%s",
            row.get("quartile"),
            row.get("bid_min"),
            row.get("bid_max"),
            row.get("MAE"),
            row.get("MdAPE"),
            row.get("n_test"),
        )
    return metrics


def load_bundle(models_dir: Path) -> dict[str, Any]:
    from model.infer import load_bundle as _load
    return _load(models_dir)


def predict_row(bundle: dict[str, Any], feature_row: dict[str, float]) -> dict[str, Any]:
    from model.infer import predict_row as _predict
    return _predict(bundle, feature_row)
