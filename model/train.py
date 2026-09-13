"""Multi-model bake-off on log1p(winning_bid).

Trains several regressors on the same stratified sold-auction split, ranks them
by price-scale MAE (then R2_price, MedAE), and persists every living candidate
plus a champion pointer for inference.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Ridge
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
    yp = np.clip(np.expm1(y_pred_log), 0, None)
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
    for k in range(n_bins, 1, -1):
        try:
            cats = pd.qcut(y, q=k, labels=False, duplicates="drop")
            return np.asarray(cats, dtype=int)
        except ValueError:
            continue
    return np.zeros(len(y), dtype=int)


def _sample_weights(y_log: np.ndarray, n_bins: int = 5) -> np.ndarray:
    bins = _log_bid_bins(y_log, n_bins=n_bins)
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


def _rank_key(m: dict[str, float]) -> tuple[float, float, float]:
    """Lower MAE wins; then higher R2; then lower MedAE."""
    mae = float(m.get("MAE") or 1e18)
    r2 = float(m.get("R2_price") if m.get("R2_price") is not None else -1e18)
    med = float(m.get("MedAE") or 1e18)
    return (mae, -r2, med)


def _fit_predict(
    name: str,
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    w_train: np.ndarray,
    supports_weight: bool,
) -> tuple[Any, np.ndarray]:
    if supports_weight:
        try:
            if isinstance(estimator, Pipeline):
                estimator.fit(X_train, y_train, model__sample_weight=w_train)
            else:
                estimator.fit(X_train, y_train, sample_weight=w_train)
        except TypeError:
            estimator.fit(X_train, y_train)
    else:
        estimator.fit(X_train, y_train)
    pred = np.asarray(estimator.predict(X_test), dtype=float)
    return estimator, pred


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    models_dir: Path,
    random_state: int = 42,
    meta: pd.DataFrame | None = None,
) -> dict[str, Any]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    n = len(X)
    bins = _log_bid_bins(y, n_bins=5)

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
    y_test_np = y_test.to_numpy()

    candidates: list[tuple[str, Any, bool, dict[str, Any]]] = []

    # --- Ridge ---
    ridge = Pipeline(
        [("scaler", StandardScaler()), ("model", Ridge(alpha=5.0))]
    )
    candidates.append(("ridge", ridge, True, {}))

    # --- ElasticNet ---
    enet = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.05, l1_ratio=0.3, max_iter=4000, random_state=random_state)),
        ]
    )
    candidates.append(("elasticnet", enet, True, {"alpha": 0.05, "l1_ratio": 0.3}))

    # --- HistGradientBoosting ---
    hgb = HistGradientBoostingRegressor(
        max_depth=6 if n >= 800 else 4,
        learning_rate=0.06,
        max_iter=250 if n >= 800 else 140,
        min_samples_leaf=max(20, n // 40),
        l2_regularization=1.0,
        random_state=random_state,
    )
    candidates.append(("hist_gbm", hgb, True, {}))

    # --- RandomForest ---
    rf = RandomForestRegressor(
        n_estimators=220 if n >= 800 else 120,
        max_depth=12 if n >= 800 else 8,
        min_samples_leaf=max(3, n // 200),
        n_jobs=-1,
        random_state=random_state,
    )
    candidates.append(("random_forest", rf, True, {}))

    # --- ExtraTrees ---
    et = ExtraTreesRegressor(
        n_estimators=250 if n >= 800 else 140,
        max_depth=14 if n >= 800 else 10,
        min_samples_leaf=max(2, n // 250),
        n_jobs=-1,
        random_state=random_state,
    )
    candidates.append(("extra_trees", et, True, {}))

    # --- LightGBM (optional) ---
    lgbm_hparams: dict[str, Any] = {}
    try:
        import lightgbm as lgb

        min_child = max(25, n // 15)
        num_leaves = 15 if n < 800 else 31
        n_estimators = 120 if n < 800 else 250
        max_depth = 4 if n < 800 else 6
        lgbm_hparams = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child,
        }
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
        candidates.append(("lightgbm", lgbm, True, lgbm_hparams))
    except Exception as exc:  # noqa: BLE001
        log.warning("LightGBM unavailable (%s) — bake-off continues without it", exc)

    fitted: dict[str, Any] = {}
    preds: dict[str, np.ndarray] = {}
    per_model: dict[str, dict[str, Any]] = {}
    leaderboard: list[dict[str, Any]] = []

    for name, est, supports_w, hp in candidates:
        try:
            model, pred = _fit_predict(
                name, est, X_train, y_train, X_test, w_train, supports_w
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Model %s failed: %s", name, exc)
            continue
        m = _metrics(y_test_np, pred)
        slices = _slice_metrics(y_test_np, pred, meta_test)
        fitted[name] = model
        preds[name] = pred
        per_model[name] = {**m, "slices": slices, "hyperparams": hp}
        leaderboard.append({"model": name, **m})
        log.info(
            "%-14s MAE=%.1f MedAE=%.1f MAPE=%.1f%% R2_price=%.3f",
            name,
            m["MAE"],
            m["MedAE"],
            m["MAPE"],
            m["R2_price"],
        )

    if not leaderboard:
        raise RuntimeError("All bake-off models failed to train")

    leaderboard.sort(key=lambda row: _rank_key(row))
    for i, row in enumerate(leaderboard, start=1):
        row["rank"] = i
    champion = leaderboard[0]["model"]
    log.info("Champion: %s (MAE=%.1f)", champion, leaderboard[0]["MAE"])

    # Feature importance from champion when available
    champ_model = fitted[champion]
    importance = pd.DataFrame({"feature": list(X.columns), "importance": 0.0})
    if hasattr(champ_model, "feature_importances_"):
        importance["importance"] = champ_model.feature_importances_
    elif isinstance(champ_model, Pipeline) and hasattr(
        champ_model.named_steps.get("model"), "coef_"
    ):
        importance["importance"] = np.abs(champ_model.named_steps["model"].coef_)
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    importance.to_csv(models_dir / "feature_importance.csv", index=False)

    if "ridge" in fitted and isinstance(fitted["ridge"], Pipeline):
        coef = pd.DataFrame(
            {
                "feature": X.columns,
                "coefficient": fitted["ridge"].named_steps["model"].coef_,
            }
        ).sort_values("coefficient", key=np.abs, ascending=False)
        coef.to_csv(models_dir / "ridge_coefficients.csv", index=False)

    # Living models: keep all fitted candidates
    living_dir = models_dir / "living"
    living_dir.mkdir(parents=True, exist_ok=True)
    for name, model in fitted.items():
        joblib.dump(model, living_dir / f"{name}.joblib")

    bundle = {
        "models": fitted,
        "champion": champion,
        "leaderboard": leaderboard,
        # Backward-compatible aliases
        "lgbm": fitted.get("lightgbm"),
        "ridge": fitted.get("ridge"),
        "feature_columns": list(X.columns),
        "median_bid": float(np.expm1(y.median())),
        "train_config": {
            "split_mode": split_mode,
            "sample_weights": "inverse_log_bid_bin_freq",
            "bakeoff": [c[0] for c in candidates],
            "champion": champion,
            "lgbm": lgbm_hparams,
        },
    }
    joblib.dump(bundle, models_dir / "model_bundle.joblib")

    # Vercel-friendly slim: prefer sklearn champion, else ridge
    slim_name = champion if champion != "lightgbm" else (
        "hist_gbm" if "hist_gbm" in fitted else "ridge"
    )
    if slim_name not in fitted:
        slim_name = "ridge" if "ridge" in fitted else next(iter(fitted))
    slim = {
        "models": {slim_name: fitted[slim_name]},
        "champion": slim_name,
        "leaderboard": [r for r in leaderboard if r["model"] == slim_name],
        "lgbm": None,
        "ridge": fitted.get("ridge") or fitted[slim_name],
        "feature_columns": list(X.columns),
        "median_bid": float(np.expm1(y.median())),
        "train_config": {
            "split_mode": split_mode,
            "vercel_slim": True,
            "primary": slim_name,
            "champion": slim_name,
        },
    }
    joblib.dump(slim, models_dir / "model_bundle_vercel.joblib", compress=3)

    (models_dir / "leaderboard.json").write_text(
        json.dumps(
            {
                "champion": champion,
                "leaderboard": leaderboard,
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "split_mode": split_mode,
            },
            indent=2,
        )
    )

    metrics: dict[str, Any] = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "n_total": int(n),
        "target": "log1p(winning_bid)",
        "split_mode": split_mode,
        "sample_weights": "inverse_log_bid_bin_freq",
        "champion": champion,
        "leaderboard": leaderboard,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "slices"} for k, v in per_model.items()},
        "lgbm_hyperparams": lgbm_hparams,
        "top_features": importance.head(20).to_dict(orient="records"),
    }
    # Compat fields for dashboard that expects lightgbm/ridge blocks
    if "lightgbm" in per_model:
        metrics["lightgbm"] = per_model["lightgbm"]
    else:
        metrics["lightgbm"] = {**per_model[champion], "note": f"compat alias → {champion}"}
    if "ridge" in per_model:
        metrics["ridge"] = per_model["ridge"]

    # Primary "live" metrics = champion
    metrics["champion_metrics"] = per_model[champion]
    (models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def load_bundle(models_dir: Path) -> dict[str, Any]:
    from model.infer import load_bundle as _load

    return _load(models_dir)


def predict_row(bundle: dict[str, Any], feature_row: dict[str, float]) -> dict[str, Any]:
    from model.infer import predict_row as _predict

    return _predict(bundle, feature_row)
