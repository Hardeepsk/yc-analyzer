"""YC Analyzer - ML model training pipeline."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import numpy as np
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler

from yc_analyzer.config import settings
from yc_analyzer.data.database import Database, get_db
from yc_analyzer.models.labeling import get_labeled_training_data, get_holdout_data


# Feature columns used for training (numeric/boolean only)
FEATURE_COLS = [
    # Base features
    "years_since_batch", "team_size", "tag_count", "location_count",
    "has_website", "is_top_company", "is_nonprofit", "is_hiring",
    "batch_size", "batch_survival_rate", "batch_unicorn_count",
    "batch_exit_count", "batch_avg_team_size",
    "industry_company_count", "industry_exit_rate",
    "fed_funds_rate_at_batch", "nasdaq_return_1yr_post_batch",
    "ai_hype_index_at_batch",
    # P5.1: Interaction features
    "team_x_industry_exit", "team_x_batch_survival",
    "batch_survival_x_maturity", "industry_density_x_exit_rate",
    "tags_x_team", "location_x_industry_exit",
    "unicorn_density_x_maturity",
    # P5.1: Polynomial features
    "team_size_sq", "years_since_batch_sq", "batch_survival_sq",
    # P5.1: Ratio features
    "team_dominance_ratio", "batch_unicorn_density", "batch_exit_density",
    # P5.1: Binary interaction flags
    "large_team_hot_industry", "small_team_strong_batch", "diverse_tags_large_batch",
]


def _prepare_xy(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X and label vector y, computing interactions from base features."""
    # Base feature columns (stored in DB)
    BASE_COLS = [
        "years_since_batch", "team_size", "tag_count", "location_count",
        "has_website", "is_top_company", "is_nonprofit", "is_hiring",
        "batch_size", "batch_survival_rate", "batch_unicorn_count",
        "batch_exit_count", "batch_avg_team_size",
        "industry_company_count", "industry_exit_rate",
        "fed_funds_rate_at_batch", "nasdaq_return_1yr_post_batch",
        "ai_hype_index_at_batch",
    ]

    # Select base columns that exist
    cols = [c for c in BASE_COLS if c in df.columns]
    base = df.select(cols).to_numpy().astype(np.float32)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)

    # Unpack for interaction computation
    years = base[:, 0] if len(cols) > 0 else np.zeros(len(df))
    team = base[:, 1] if len(cols) > 1 else np.zeros(len(df))
    tags = base[:, 2] if len(cols) > 2 else np.zeros(len(df))
    locs = base[:, 3] if len(cols) > 3 else np.zeros(len(df))
    bsize = base[:, 8] if len(cols) > 8 else np.zeros(len(df))
    bsurv = base[:, 9] if len(cols) > 9 else np.zeros(len(df))
    buni = base[:, 10] if len(cols) > 10 else np.zeros(len(df))
    bexit = base[:, 11] if len(cols) > 11 else np.zeros(len(df))
    ico_count = base[:, 13] if len(cols) > 13 else np.zeros(len(df))
    ico_rate = base[:, 14] if len(cols) > 14 else np.zeros(len(df))

    # Compute interactions (must match predict.py exactly)
    interactions = np.column_stack([
        team * ico_rate,                        # team_x_industry_exit
        team * bsurv,                           # team_x_batch_survival
        bsurv * years,                          # batch_survival_x_maturity
        ico_count * ico_rate,                   # industry_density_x_exit_rate
        tags * team,                            # tags_x_team
        locs * ico_rate,                        # location_x_industry_exit
        buni * years,                           # unicorn_density_x_maturity
        team ** 2,                              # team_size_sq
        years ** 2,                             # years_since_batch_sq
        bsurv ** 2,                             # batch_survival_sq
        np.where(bsize > 0, team / bsize, 0.0),          # team_dominance_ratio
        np.where(bsize > 0, buni / bsize, 0.0),          # batch_unicorn_density
        np.where(bsize > 0, bexit / bsize, 0.0),         # batch_exit_density
        ((team > 10) & (ico_rate > 0.1)).astype(float),  # large_team_hot_industry
        ((team <= 5) & (bsurv > 0.5)).astype(float),     # small_team_strong_batch
        ((tags > 3) & (bsize > 100)).astype(float),      # diverse_tags_large_batch
    ])

    X = np.hstack([base, interactions]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df.select("success_at_5yr").to_numpy().astype(np.float32).ravel()

    return X, y, FEATURE_COLS


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train a baseline LogisticRegression."""
    logger.info("Training baseline LogisticRegression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=settings.random_seed)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train XGBoost classifier."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed, skipping")
        return None

    logger.info("Training XGBoost...")

    # Compute scale_pos_weight for imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": float(scale),
        "seed": settings.random_seed,
        "verbosity": 0,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(
        params, dtrain, num_boost_round=300,
        verbose_eval=False,
    )
    return model


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train LightGBM classifier."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed, skipping")
        return None

    logger.info("Training LightGBM...")

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "scale_pos_weight": scale,
        "seed": settings.random_seed,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=300)
    return model


def evaluate_model(
    model: Any, scaler: Optional[Any], X_test: np.ndarray, y_test: np.ndarray,
    model_name: str, feature_names: List[str], xgb_model: bool = False,
) -> Dict[str, Any]:
    """Evaluate a model and return metrics."""
    if model is None:
        return {"error": "model not trained"}

    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    # Get probabilities
    if xgb_model:
        try:
            import xgboost as xgb
            dtest = xgb.DMatrix(X_eval)
            y_prob = model.predict(dtest)
        except Exception:
            y_prob = model.predict(X_eval)
    else:
        y_prob = model.predict_proba(X_eval)[:, 1]

    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    metrics = {"model_name": model_name}
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_prob))
    except Exception:
        metrics["auc_roc"] = 0.5

    try:
        metrics["auc_pr"] = float(average_precision_score(y_test, y_prob))
    except Exception:
        metrics["auc_pr"] = 0.5

    metrics["brier_score"] = float(brier_score_loss(y_test, y_prob))
    metrics["classification_report"] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    # Precision@k (top 10% predicted)
    k = max(1, int(len(y_test) * 0.1))
    top_k_idx = np.argsort(y_prob)[-k:]
    metrics["precision_at_10pct"] = float(y_test[top_k_idx].mean())

    # Calibration
    try:
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=5)
        metrics["calibration"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
    except Exception:
        metrics["calibration"] = {}

    return metrics


def run_training_pipeline(db: Optional[Database] = None) -> Dict[str, Any]:
    """Run the full training pipeline."""
    db = db or get_db()
    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df, test_df = get_holdout_data(cutoff_year=2021, db=db)

    if len(train_df) == 0:
        logger.error("No training data available")
        return {"error": "no training data"}

    X_train, y_train, feature_names = _prepare_xy(train_df)
    X_test, y_test, _ = _prepare_xy(test_df)

    logger.info(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    logger.info(f"Positive rate train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")

    all_metrics = {}

    # Baseline
    lr_model, lr_scaler = train_baseline(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, lr_scaler, X_test, y_test, "LogisticRegression", feature_names)
    all_metrics["logistic_regression"] = lr_metrics
    logger.info(f"LogisticRegression AUC: {lr_metrics.get('auc_roc', 'N/A')}")

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    if xgb_model is not None:
        xgb_metrics = evaluate_model(xgb_model, None, X_test, y_test, "XGBoost", feature_names, xgb_model=True)
        all_metrics["xgboost"] = xgb_metrics
        logger.info(f"XGBoost AUC: {xgb_metrics.get('auc_roc', 'N/A')}")

        # Save model
        xgb_model.save_model(str(model_dir / "xgb_success_v1.json"))

        # Feature importance
        importance = dict(zip(feature_names, xgb_model.get_fscore().values()))
        importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        with open(model_dir / "feature_importance_xgb.json", "w") as f:
            json.dump(importance_sorted, f, indent=2)

    # LightGBM
    lgb_model = train_lightgbm(X_train, y_train)
    if lgb_model is not None:
        lgb_metrics = evaluate_model(lgb_model, None, X_test, y_test, "LightGBM", feature_names, xgb_model=True)
        all_metrics["lightgbm"] = lgb_metrics
        logger.info(f"LightGBM AUC: {lgb_metrics.get('auc_roc', 'N/A')}")

        # Save model
        lgb_model.save_model(str(model_dir / "lgb_success_v1.txt"))

        # Feature importance
        lgb_imp = dict(zip(feature_names, lgb_model.feature_importance(importance_type="gain")))
        lgb_imp_sorted = dict(sorted(lgb_imp.items(), key=lambda x: x[1], reverse=True))
        with open(model_dir / "feature_importance_lgb.json", "w") as f:
            json.dump(lgb_imp_sorted, f, indent=2)

    # Save metrics
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info(f"Models and metrics saved to {model_dir}")
    return all_metrics


if __name__ == "__main__":
    run_training_pipeline()
