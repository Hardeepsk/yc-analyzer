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
    # P5.4: Tag rarity features
    "tag_idf_score", "tag_uniqueness", "tag_trending_score",
]


def _prepare_xy(df: pl.DataFrame, db: Optional[Database] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract feature matrix X and label vector y, computing interactions from base features."""
    db = db or get_db()

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

    # P5.4: Tag rarity features (computed from companies.tags)
    tag_features = _compute_tag_features(df, db)
    X = np.hstack([base, interactions, tag_features]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df.select("success_at_5yr").to_numpy().astype(np.float32).ravel()

    return X, y, FEATURE_COLS


def _compute_tag_features(df: pl.DataFrame, db: Database) -> np.ndarray:
    """Compute tag IDF, uniqueness, and trending score (P5.4)."""
    logger.info("Computing tag features...")

    # Get all tags with company IDs
    tag_rows = db.conn.execute("""
        SELECT id, tags FROM companies WHERE tags IS NOT NULL AND len(tags) > 0
    """).fetchall()

    if not tag_rows:
        n = len(df)
        return np.zeros((n, 3), dtype=np.float32)

    # Build tag frequency map (IDF)
    from collections import Counter
    tag_doc_freq = Counter()
    company_tags = {}
    for cid, tag_list in tag_rows:
        if tag_list is None:
            continue
        unique_tags = list(set(tag_list))  # deduplicate per company
        company_tags[cid] = unique_tags
        for t in unique_tags:
            tag_doc_freq[t] += 1

    n_companies = len(tag_rows)
    # IDF = log(N / df(t)) — rare tags get higher weight
    tag_idf = {t: np.log(n_companies / max(freq, 1)) for t, freq in tag_doc_freq.items()}

    # Compute trending score: tags appearing more in recent batches (2023+)
    recent_tags = Counter()
    older_tags = Counter()
    for cid, tag_list in company_tags.items():
        # Look up batch year for this company
        row = db.conn.execute("SELECT batch FROM companies WHERE id = ?", [cid]).fetchone()
        if row and row[0]:
            import re
            m = re.search(r"(\d{4})", row[0])
            if m:
                year = int(m.group(1))
                for t in tag_list:
                    if year >= 2023:
                        recent_tags[t] += 1
                    else:
                        older_tags[t] += 1

    # Trending = recent_freq / max(older_freq, 1)
    tag_trending = {}
    all_tags = set(list(recent_tags.keys()) + list(older_tags.keys()))
    for t in all_tags:
        recent = recent_tags.get(t, 0)
        older = older_tags.get(t, 1)
        tag_trending[t] = recent / max(older, 1)

    # Build feature vectors for each company in df
    company_ids = df["id"].to_list() if "id" in df.columns else []
    result = np.zeros((len(df), 3), dtype=np.float32)

    for i, cid in enumerate(company_ids):
        tag_list = company_tags.get(cid, [])
        if not tag_list:
            continue

        # tag_idf_score: average IDF of company's tags
        idf_vals = [tag_idf.get(t, 0.0) for t in tag_list]
        result[i, 0] = np.mean(idf_vals) if idf_vals else 0.0

        # tag_uniqueness: ratio of rare tags (IDF > median)
        median_idf = np.median(list(tag_idf.values())) if tag_idf else 0.0
        rare_count = sum(1 for v in idf_vals if v > median_idf)
        result[i, 1] = rare_count / len(tag_list) if tag_list else 0.0

        # tag_trending_score: average trending score of company's tags
        trending_vals = [tag_trending.get(t, 1.0) for t in tag_list]
        result[i, 2] = np.mean(trending_vals) if trending_vals else 1.0

    return result


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

        # P7.1: Feature selection — drop features with <1% importance
        selected_indices, selected_names, selected_X_train, selected_X_test = _select_features(
            importance_sorted, feature_names, X_train, X_test
        )

        if len(selected_names) < len(feature_names):
            logger.info(f"P7.1: Selected {len(selected_names)}/{len(feature_names)} features")
            logger.info(f"P7.1: Dropped: {[n for n in feature_names if n not in selected_names]}")

            # Retrain XGBoost on selected features
            xgb_selected = train_xgboost(selected_X_train, y_train)
            if xgb_selected is not None:
                xgb_sel_metrics = evaluate_model(xgb_selected, None, selected_X_test, y_test, "XGBoost_Selected", selected_names, xgb_model=True)
                all_metrics["xgboost_selected"] = xgb_sel_metrics
                logger.info(f"XGBoost (selected) AUC: {xgb_sel_metrics.get('auc_roc', 'N/A')}")

                # Save if better
                if xgb_sel_metrics.get('auc_roc', 0) > xgb_metrics.get('auc_roc', 0):
                    xgb_selected.save_model(str(model_dir / "xgb_success_v1.json"))
                    logger.info("P7.1: Selected features model is BETTER — saved as primary")
                    # Update feature importance
                    imp_sel = dict(zip(selected_names, xgb_selected.get_fscore().values()))
                    imp_sel_sorted = dict(sorted(imp_sel.items(), key=lambda x: x[1], reverse=True))
                    with open(model_dir / "feature_importance_xgb.json", "w") as f:
                        json.dump(imp_sel_sorted, f, indent=2)
                    # Save selected feature names
                    with open(model_dir / "selected_features.json", "w") as f:
                        json.dump(selected_names, f, indent=2)

        # P7.5: Pseudo-labeling — expand training set with high-confidence predictions
        xgb_pseudo, y_pseudo = _pseudo_label(xgb_model, db, feature_names)
        if len(y_pseudo) > 0:
            # Combine original + pseudo-labeled data
            X_train_aug = np.vstack([X_train, xgb_pseudo])
            y_train_aug = np.concatenate([y_train, y_pseudo])

            # Assign sample weights: 1.0 for real, 0.5 for pseudo
            sample_weights = np.concatenate([
                np.ones(len(y_train)),
                np.full(len(y_pseudo), 0.5),
            ])

            logger.info(f"P7.5: Training with {len(y_train)} real + {len(y_pseudo)} pseudo = {len(y_train_aug)} total")

            # Retrain XGBoost with sample weights
            try:
                import xgboost as xgb_lib
                dtrain_aug = xgb_lib.DMatrix(X_train_aug, label=y_train_aug, weight=sample_weights)
                n_neg = (y_train_aug == 0).sum()
                n_pos = (y_train_aug == 1).sum()
                scale = n_neg / max(n_pos, 1)
                params = {
                    "objective": "binary:logistic", "eval_metric": "auc",
                    "max_depth": 5, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "scale_pos_weight": float(scale), "seed": 42, "verbosity": 0,
                }
                xgb_pseudo_model = xgb_lib.train(params, dtrain_aug, num_boost_round=300)
                pseudo_metrics = evaluate_model(xgb_pseudo_model, None, X_test, y_test, "XGBoost_Pseudo", feature_names, xgb_model=True)
                all_metrics["xgboost_pseudo"] = pseudo_metrics
                logger.info(f"XGBoost (pseudo-labeled) AUC: {pseudo_metrics.get('auc_roc', 'N/A')}")

                # Save if better
                best_auc = max(
                    all_metrics.get("xgboost", {}).get("auc_roc", 0),
                    all_metrics.get("xgboost_selected", {}).get("auc_roc", 0),
                )
                if pseudo_metrics.get("auc_roc", 0) > best_auc:
                    xgb_pseudo_model.save_model(str(model_dir / "xgb_success_v1.json"))
                    logger.info("P7.5: Pseudo-labeled model is BEST — saved as primary")
            except Exception as e:
                logger.warning(f"P7.5: Pseudo-labeling failed: {e}")

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


def _pseudo_label(
    xgb_model: Any,
    db: Database,
    feature_names: List[str],
    confidence_threshold: float = 0.80,
) -> Tuple[np.ndarray, np.ndarray]:
    """P7.5: Generate pseudo-labels for unlabeled companies.

    Uses the trained XGBoost model to predict on companies without5yr labels,
    then adds high-confidence predictions (>80% or <20%) as training data.
    Returns (X_pseudo, y_pseudo).
    """
    logger.info("P7.5: Generating pseudo-labels for unlabeled companies...")

    # Get unlabeled companies (no success_at_5yr label, not censored)
    unlabeled_df = pl.from_arrow(db.conn.execute("""
        SELECT c.id
        FROM companies c
        LEFT JOIN companies_enriched ce ON c.id = ce.company_id
        WHERE ce.success_at_5yr IS NULL
          AND (ce.is_censored = FALSE OR ce.is_censored IS NULL)
    """).arrow())

    if len(unlabeled_df) == 0:
        logger.info("P7.5: No unlabeled companies found")
        return np.array([]), np.array([])

    # Prepare features for unlabeled companies
    unlabeled_ids = unlabeled_df["id"].to_list()

    # We need to prepare features using the same pipeline
    # Build a mini DataFrame with all the enriched columns
    placeholders = pl.DataFrame({"id": unlabeled_ids})

    # Join with enriched to get base features
    enriched_query = """
        SELECT ce.*, c.batch, c.tags, c.industry
        FROM companies_enriched ce
        JOIN companies c ON c.id = ce.company_id
        WHERE ce.company_id IN ({})
    """.format(",".join(str(i) for i in unlabeled_ids))

    try:
        unlabeled_full = pl.from_arrow(db.conn.execute(enriched_query).arrow())
    except Exception as e:
        logger.warning(f"P7.5: Could not load unlabeled data: {e}")
        return np.array([]), np.array([])

    # Prepare features using same pipeline
    X_unlabeled, _, _ = _prepare_xy(unlabeled_full, db)

    # Get predictions
    import xgboost as xgb_lib
    dmat = xgb_lib.DMatrix(X_unlabeled)
    probs = xgb_model.predict(dmat)

    # Select high-confidence predictions
    high_pos = probs >= confidence_threshold
    high_neg = probs <= (1.0 - confidence_threshold)

    X_pseudo_list = []
    y_pseudo_list = []

    if high_pos.any():
        X_pseudo_list.append(X_unlabeled[high_pos])
        y_pseudo_list.append(np.ones(high_pos.sum()))
        logger.info(f"P7.5: {high_pos.sum()} high-confidence positive pseudo-labels")

    if high_neg.any():
        X_pseudo_list.append(X_unlabeled[high_neg])
        y_pseudo_list.append(np.zeros(high_neg.sum()))
        logger.info(f"P7.5: {high_neg.sum()} high-confidence negative pseudo-labels")

    if not X_pseudo_list:
        return np.array([]), np.array([])

    X_pseudo = np.vstack(X_pseudo_list)
    y_pseudo = np.concatenate(y_pseudo_list)

    logger.info(f"P7.5: Total pseudo-labeled: {len(y_pseudo)} companies")
    return X_pseudo, y_pseudo


def _select_features(
    importance_sorted: Dict[str, float],
    feature_names: List[str],
    X_train: np.ndarray,
    X_test: np.ndarray,
    threshold_pct: float = 0.01,
) -> Tuple[List[int], List[str], np.ndarray, np.ndarray]:
    """Select features with importance above threshold percentage of max.

    Returns (indices, names, filtered_X_train, filtered_X_test).
    """
    if not importance_sorted:
        return list(range(len(feature_names))), feature_names, X_train, X_test

    max_imp = max(importance_sorted.values())
    if max_imp <= 0:
        return list(range(len(feature_names))), feature_names, X_train, X_test

    # Keep features with importance > threshold% of max
    min_importance = max_imp * threshold_pct
    selected = [name for name, imp in importance_sorted.items() if imp >= min_importance]

    # Always keep at least top-10 features
    if len(selected) < 10:
        selected = list(importance_sorted.keys())[:10]

    # Map back to indices
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    indices = [name_to_idx[name] for name in selected if name in name_to_idx]

    return indices, selected, X_train[:, indices], X_test[:, indices]


if __name__ == "__main__":
    run_training_pipeline()
