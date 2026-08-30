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
    "years_since_batch", "team_size", "tag_count", "location_count",
    "has_website", "is_top_company", "is_nonprofit", "is_hiring",
    "batch_size", "batch_survival_rate", "batch_unicorn_count",
    "batch_exit_count", "batch_avg_team_size",
    "industry_company_count", "industry_exit_rate",
    "fed_funds_rate_at_batch", "nasdaq_return_1yr_post_batch",
    "ai_hype_index_at_batch",
]


def _prepare_xy(df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X and label vector y from polars DataFrame."""
    cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df.select(cols).to_numpy().astype(np.float32)
    y = df.select("success_at_5yr").to_numpy().astype(np.float32).ravel()

    # Fill NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, cols


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
