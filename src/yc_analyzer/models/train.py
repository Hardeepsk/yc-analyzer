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

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed, explanations disabled")

from yc_analyzer.config import settings
from yc_analyzer.data.database import Database, get_db
from yc_analyzer.models.labeling import get_labeled_training_data, get_holdout_data
from yc_analyzer.features.engineering import NLPEmbedder, get_nlp_embedder


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
    # P8: Founder features (scraped from the accelerator API)
    "founder_count", "has_technical_founder", "has_repeat_founder",
    "founder_linkedin_count", "max_founder_bio_length",
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
    # P5.3: Geographic density features
    "hub_unicorn_count", "hub_company_count", "is_in_hub",
    # P5.5: Batch momentum features
    "batch_unicorn_growth", "batch_exit_growth", "batch_size_growth",
    "batch_survival_trend", "batch_momentum_score",
    # P8: NLP embedding features (4 fields x 16 PCA components = 64)
    # desc = long_description, short = short_description, tags = concatenated tags,
    # ind = concatenated industries (see NLPEmbedder in features/engineering.py)
    "nlp_desc_0", "nlp_desc_1", "nlp_desc_2", "nlp_desc_3", "nlp_desc_4", "nlp_desc_5",
    "nlp_desc_6", "nlp_desc_7", "nlp_desc_8", "nlp_desc_9", "nlp_desc_10", "nlp_desc_11",
    "nlp_desc_12", "nlp_desc_13", "nlp_desc_14", "nlp_desc_15",
    "nlp_short_0", "nlp_short_1", "nlp_short_2", "nlp_short_3", "nlp_short_4", "nlp_short_5",
    "nlp_short_6", "nlp_short_7", "nlp_short_8", "nlp_short_9", "nlp_short_10", "nlp_short_11",
    "nlp_short_12", "nlp_short_13", "nlp_short_14", "nlp_short_15",
    "nlp_tags_0", "nlp_tags_1", "nlp_tags_2", "nlp_tags_3", "nlp_tags_4", "nlp_tags_5",
    "nlp_tags_6", "nlp_tags_7", "nlp_tags_8", "nlp_tags_9", "nlp_tags_10", "nlp_tags_11",
    "nlp_tags_12", "nlp_tags_13", "nlp_tags_14", "nlp_tags_15",
    "nlp_ind_0", "nlp_ind_1", "nlp_ind_2", "nlp_ind_3", "nlp_ind_4", "nlp_ind_5",
    "nlp_ind_6", "nlp_ind_7", "nlp_ind_8", "nlp_ind_9", "nlp_ind_10", "nlp_ind_11",
    "nlp_ind_12", "nlp_ind_13", "nlp_ind_14", "nlp_ind_15",
    # P1.1: Funding features
    "has_funding_data", "total_raised_usd", "last_valuation_usd",
    "round_count", "funding_stage_encoded", "years_since_last_round", "investor_quality_score",
]

# NLP feature column names (mirrors NLPEmbedder.column_names) for reuse in predict.py
NLP_FEATURE_COLS = NLPEmbedder().column_names


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
        # P8: Founder features (scraped from the accelerator API)
        "founder_count", "has_technical_founder", "has_repeat_founder",
        "founder_linkedin_count", "max_founder_bio_length",
        # P1.1: Funding features
        "has_funding_data", "total_raised_usd", "last_valuation_usd",
        "round_count", "funding_stage_encoded", "years_since_last_round", "investor_quality_score",
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

    # P5.4: Tag rarity features
    tag_features = _compute_tag_features(df, db)
    # P5.3: Geographic density features
    geo_features = _compute_geo_features(df, db)
    # P5.5: Batch momentum features
    momentum_features = _compute_momentum_features(df, db)

    # P8: NLP embedding features (sentence-transformers + PCA, or TF-IDF + PCA fallback)
    nlp_features = _compute_nlp_features(df, db)

    X = np.hstack([
        base, interactions, tag_features, geo_features, momentum_features, nlp_features
    ]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y = df.select("success_at_5yr").to_numpy().astype(np.float32).ravel()

    return X, y, FEATURE_COLS


def _compute_nlp_features(df: pl.DataFrame, db: Database) -> np.ndarray:
    """Compute NLP embedding features for the companies in ``df`` (P8).

    Uses a fitted ``NLPEmbedder`` (fit on the first dataframe seen, typically the
    training split, and cached in-process). Persists the fitted transformers and a
    cached embedding matrix to ``models/`` for prediction time.
    """
    ids = _extract_ids_for_nlp(df)
    if not ids:
        logger.warning("No company ids found for NLP features; returning zeros")
        return np.zeros((len(df), len(NLP_FEATURE_COLS)), dtype=np.float32)
    try:
        embedder = get_nlp_embedder(df, db)
        feats = embedder.transform(ids, db)
        if feats.shape[0] != len(df):
            # Defensive: align by id order if shapes diverge
            feats = np.zeros((len(df), len(NLP_FEATURE_COLS)), dtype=np.float32)
        return feats.astype(np.float32)
    except Exception as e:
        logger.warning(f"NLP feature computation failed ({e}); using zeros")
        return np.zeros((len(df), len(NLP_FEATURE_COLS)), dtype=np.float32)


def _extract_ids_for_nlp(df: pl.DataFrame) -> List[int]:
    """Extract company ids from a training/holdout DataFrame (id or company_id)."""
    if "company_id" in df.columns:
        return [int(x) for x in df["company_id"].to_list()]
    if "id" in df.columns:
        return [int(x) for x in df["id"].to_list()]
    return []


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
    company_ids = df["company_id"].to_list() if "company_id" in df.columns else []
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


def _compute_geo_features(df: pl.DataFrame, db: Database) -> np.ndarray:
    """Compute geographic density features (P5.3).

    - hub_unicorn_count: number of unicorns in the company's primary location
    - hub_company_count: total companies in that location
    - is_in_hub: binary flag for top-10 unicorn hubs
    """
    logger.info("Computing geographic features...")

    n = len(df)
    result = np.zeros((n, 3), dtype=np.float32)

    # Get all locations with unicorn counts using CTE with UNNEST
    try:
        location_stats = db.conn.execute("""
            WITH unnested AS (
                SELECT
                    UNNEST(all_locations) AS location,
                    top_company
                FROM companies
                WHERE all_locations IS NOT NULL AND len(all_locations) > 0
            )
            SELECT
                location,
                SUM(CASE WHEN top_company THEN 1 ELSE 0 END) AS unicorns,
                COUNT(*) AS total
            FROM unnested
            GROUP BY location
        """).fetchall()
    except Exception:
        location_stats = []

    loc_unicorn = {r[0]: r[1] for r in location_stats}
    loc_total = {r[0]: r[2] for r in location_stats}

    # Top-10 hubs by unicorn count
    top_hubs = sorted(loc_unicorn.items(), key=lambda x: x[1], reverse=True)[:10]
    hub_set = set(h for h, _ in top_hubs)

    # For each company, get primary location
    company_ids = df["company_id"].to_list() if "company_id" in df.columns else []
    for i, cid in enumerate(company_ids):
        row = db.conn.execute(
            "SELECT all_locations FROM companies WHERE id = ?", [cid]
        ).fetchone()
        if not row or not row[0]:
            continue

        locations = row[0]
        if not locations:
            continue

        primary_loc = locations[0] if isinstance(locations, list) else locations

        result[i, 0] = float(loc_unicorn.get(primary_loc, 0))
        result[i, 1] = float(loc_total.get(primary_loc, 0))
        result[i, 2] = 1.0 if primary_loc in hub_set else 0.0

    return result


def _compute_momentum_features(df: pl.DataFrame, db: Database) -> np.ndarray:
    """Compute batch momentum features (P5.5).

    - batch_unicorn_growth: change in unicorn density vs previous batch
    - batch_exit_growth: change in exit rate vs previous batch
    - batch_size_growth: change in batch size vs previous batch
    - batch_survival_trend: trend in survival rate over last 3 batches
    - batch_momentum_score: composite momentum score
    """
    logger.info("Computing batch momentum features...")

    n = len(df)
    result = np.zeros((n, 5), dtype=np.float32)

    # Get batch stats ordered by batch
    batch_rows = db.conn.execute("""
        SELECT batch, company_count, survival_rate, unicorn_count, exit_count
        FROM batches
        ORDER BY batch
    """).fetchall()

    if len(batch_rows) < 2:
        return result

    # Build batch index and compute deltas
    batch_list = [r[0] for r in batch_rows]
    unicorn_counts = [r[3] for r in batch_rows]
    exit_counts = [r[4] for r in batch_rows]
    company_counts = [r[1] for r in batch_rows]
    survival_rates = [r[2] for r in batch_rows]

    # Compute per-batch growth metrics
    batch_unicorn_growth = {}
    batch_exit_growth = {}
    batch_size_growth = {}
    batch_survival_trend = {}

    for i in range(1, len(batch_rows)):
        prev_companies = company_counts[i-1] if company_counts[i-1] > 0 else 1
        curr_companies = company_counts[i] if company_counts[i] > 0 else 1

        # Unicorn density growth
        prev_density = unicorn_counts[i-1] / prev_companies
        curr_density = unicorn_counts[i] / curr_companies
        batch_unicorn_growth[batch_list[i]] = curr_density - prev_density

        # Exit rate growth
        prev_exit_rate = exit_counts[i-1] / prev_companies
        curr_exit_rate = exit_counts[i] / curr_companies
        batch_exit_growth[batch_list[i]] = curr_exit_rate - prev_exit_rate

        # Size growth
        batch_size_growth[batch_list[i]] = (curr_companies - prev_companies) / prev_companies

        # Survival trend (3-batch)
        if i >= 3:
            batch_survival_trend[batch_list[i]] = (
                survival_rates[i] - survival_rates[i-3]
            ) / 3.0

    # Map to companies
    company_ids = df["company_id"].to_list() if "company_id" in df.columns else []
    company_batches = df["batch"].to_list() if "batch" in df.columns else []

    for i, (cid, batch) in enumerate(zip(company_ids, company_batches)):
        if not batch:
            continue

        result[i, 0] = batch_unicorn_growth.get(batch, 0.0)
        result[i, 1] = batch_exit_growth.get(batch, 0.0)
        result[i, 2] = batch_size_growth.get(batch, 0.0)
        result[i, 3] = batch_survival_trend.get(batch, 0.0)
        # Composite momentum score
        result[i, 4] = (
            result[i, 0] * 2 +   # unicorn growth weighted more
            result[i, 1] * 1.5 + # exit growth
            result[i, 2] * 0.5 + # size growth
            result[i, 3] * 1.0   # survival trend
        )

    return result


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train a baseline LogisticRegression."""
    logger.info("Training baseline LogisticRegression...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=settings.random_seed)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray, sample_weights: Optional[np.ndarray] = None, model_params: Optional[Dict[str, Any]] = None) -> Any:
    """Train XGBoost classifier with optional cost-sensitive weighting.

    If ``model_params`` is provided (search-space naming from Optuna tuning), the
    tuned hyperparameters are applied (max_depth, learning_rate, subsample,
    colsample_bytree, reg_alpha, reg_lambda, min_child_weight, n_estimators).
    """
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

    # P7.2: Focal loss-style weighting — downweight easy examples
    if sample_weights is None:
        sample_weights = _compute_focal_weights(X_train, y_train)

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
    num_boost_round = 300
    if model_params:
        # Apply tuned hyperparameters (search-space naming)
        for _key in ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                     "reg_alpha", "reg_lambda", "min_child_weight"):
            if _key in model_params:
                params[_key] = model_params[_key]
        if "n_estimators" in model_params:
            num_boost_round = int(model_params["n_estimators"])

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    model = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        verbose_eval=False,
    )
    return model


def _compute_focal_weights(X: np.ndarray, y: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Compute focal loss-style sample weights.

    Easy examples (predicted correctly with high confidence) get lower weight.
    Hard examples (predicted incorrectly or near decision boundary) get higher weight.
    """
    from sklearn.linear_model import LogisticRegression as LR

    # Quick calibration model to get initial probabilities
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lr = LR(max_iter=500, random_state=42)
    lr.fit(X_scaled, y)
    probs = lr.predict_proba(X_scaled)[:, 1]

    # Focal weight: (1 - p_t)^gamma where p_t = probability of true class
    p_t = np.where(y == 1, probs, 1 - probs)
    weights = (1 - p_t) ** gamma

    # Normalize to mean=1
    weights = weights / weights.mean()

    # Clip to prevent extreme weights
    weights = np.clip(weights, 0.1, 10.0)

    logger.info(f"P7.2: Focal weights — mean={weights.mean():.3f}, min={weights.min():.3f}, max={weights.max():.3f}")
    return weights


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray, model_params: Optional[Dict[str, Any]] = None) -> Any:
    """Train LightGBM classifier.

    If ``model_params`` is provided (search-space naming from Optuna tuning), the
    tuned hyperparameters are translated to LightGBM param names and applied.
    """
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
    num_boost_round = 300
    if model_params:
        # Translate search-space names to LightGBM param names
        _translation = {
            "max_depth": "max_depth",
            "learning_rate": "learning_rate",
            "colsample_bytree": "feature_fraction",
            "subsample": "bagging_fraction",
            "min_child_weight": "min_child_weight",
            "reg_alpha": "lambda_l1",
            "reg_lambda": "lambda_l2",
        }
        for _src, _dst in _translation.items():
            if _src in model_params:
                params[_dst] = model_params[_src]
        if model_params.get("subsample", 1.0) < 1.0:
            params["bagging_freq"] = 1
        if "n_estimators" in model_params:
            num_boost_round = int(model_params["n_estimators"])

    dtrain = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return model


def optimize_xgboost(
    db: Optional[Database] = None,
    n_trials: int = 50,
    n_splits: int = 3,
    use_focal: bool = True,
) -> Optional[Dict[str, Any]]:
    """Tune XGBoost hyperparameters with Optuna using temporal cross-validation.

    Uses ``TimeSeriesSplit`` over the temporal training set (batches <= 2021, sorted
    by batch year) to maximize AUC-ROC across folds. Focal weights (P7.2) are applied
    per-fold when ``use_focal`` is True, matching the production ``train_xgboost``.

    Returns the best hyperparameter dict (search-space naming) or None on failure.
    """
    try:
        import optuna
        import xgboost as xgb
    except ImportError:
        logger.warning("optuna or xgboost not installed, skipping XGBoost tuning")
        return None

    db = db or get_db()
    logger.info("Loading training data for XGBoost tuning (temporal split, cutoff 2021)...")
    train_df, _ = get_holdout_data(cutoff_year=2021, db=db)
    if len(train_df) == 0:
        logger.error("No training data available for tuning")
        return None

    # Sort temporally so TimeSeriesSplit respects time ordering
    if "batch_year" in train_df.columns:
        train_df = train_df.sort("batch_year")
    X, y, _ = _prepare_xy(train_df, db)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "seed": settings.random_seed,
            "verbosity": 0,
        }
        num_boost_round = int(trial.suggest_int("n_estimators", 100, 1000, step=50))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs: List[float] = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            if len(np.unique(y_va)) < 2:
                continue

            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            p = dict(params)
            p["scale_pos_weight"] = float(n_neg / max(n_pos, 1))

            sw = _compute_focal_weights(X_tr, y_tr) if use_focal else None
            dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=sw)
            dval = xgb.DMatrix(X_va)
            model = xgb.train(
                p, dtrain, num_boost_round=num_boost_round,
                evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False,
            )
            if getattr(model, "best_iteration", None) is not None:
                prob = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
            else:
                prob = model.predict(dval)
            aucs.append(float(roc_auc_score(y_va, prob)))

        return float(np.mean(aucs)) if aucs else 0.5

    logger.info(f"Starting XGBoost Optuna study with {n_trials} trials...")
    study = optuna.create_study(direction="maximize", study_name="xgboost_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    logger.info(f"XGBoost best CV AUC: {study.best_value:.4f}")
    logger.info(f"XGBoost best params: {best}")
    return best


def optimize_lightgbm(
    db: Optional[Database] = None,
    n_trials: int = 50,
    n_splits: int = 3,
) -> Optional[Dict[str, Any]]:
    """Tune LightGBM hyperparameters with Optuna using temporal cross-validation.

    Mirrors ``optimize_xgboost`` but translates the search-space names to LightGBM
    param names (subsample->bagging_fraction, colsample_bytree->feature_fraction,
    reg_alpha->lambda_l1, reg_lambda->lambda_l2).

    Returns the best hyperparameter dict (search-space naming) or None on failure.
    """
    try:
        import optuna
        import lightgbm as lgb
    except ImportError:
        logger.warning("optuna or lightgbm not installed, skipping LightGBM tuning")
        return None

    db = db or get_db()
    logger.info("Loading training data for LightGBM tuning (temporal split, cutoff 2021)...")
    train_df, _ = get_holdout_data(cutoff_year=2021, db=db)
    if len(train_df) == 0:
        logger.error("No training data available for tuning")
        return None

    if "batch_year" in train_df.columns:
        train_df = train_df.sort("batch_year")
    X, y, _ = _prepare_xy(train_df, db)

    def objective(trial: "optuna.Trial") -> float:
        search = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        num_boost_round = int(trial.suggest_int("n_estimators", 100, 1000, step=50))

        lgb_params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "learning_rate": search["learning_rate"],
            "feature_fraction": search["colsample_bytree"],
            "bagging_fraction": search["subsample"],
            "min_child_weight": search["min_child_weight"],
            "lambda_l1": search["reg_alpha"],
            "lambda_l2": search["reg_lambda"],
            "max_depth": search["max_depth"],
            "seed": settings.random_seed,
            "verbose": -1,
        }
        if search["subsample"] < 1.0:
            lgb_params["bagging_freq"] = 1

        tscv = TimeSeriesSplit(n_splits=n_splits)
        aucs: List[float] = []
        for tr_idx, va_idx in tscv.split(X):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            if len(np.unique(y_va)) < 2:
                continue

            n_neg = int((y_tr == 0).sum())
            n_pos = int((y_tr == 1).sum())
            p = dict(lgb_params)
            p["scale_pos_weight"] = n_neg / max(n_pos, 1)

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va)
            model = lgb.train(
                p, dtrain, num_boost_round=num_boost_round,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            it = model.best_iteration if getattr(model, "best_iteration", None) else num_boost_round
            prob = model.predict(X_va, num_iteration=it)
            aucs.append(float(roc_auc_score(y_va, prob)))

        return float(np.mean(aucs)) if aucs else 0.5

    logger.info(f"Starting LightGBM Optuna study with {n_trials} trials...")
    study = optuna.create_study(direction="maximize", study_name="lightgbm_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = dict(study.best_params)
    logger.info(f"LightGBM best CV AUC: {study.best_value:.4f}")
    logger.info(f"LightGBM best params: {best}")
    return best


def run_tuning(db: Optional[Database] = None, n_trials: int = 50) -> Dict[str, Any]:
    """Run Optuna tuning for both models and save best params to models/best_params.json.

    Returns the dict of best params (keys: ``xgboost``, ``lightgbm``).
    """
    db = db or get_db()
    settings.model_dir.mkdir(parents=True, exist_ok=True)

    best_params: Dict[str, Any] = {}
    xgb_best = optimize_xgboost(db, n_trials=n_trials)
    if xgb_best is not None:
        best_params["xgboost"] = xgb_best
    lgb_best = optimize_lightgbm(db, n_trials=n_trials)
    if lgb_best is not None:
        best_params["lightgbm"] = lgb_best

    if best_params:
        path = settings.model_dir / "best_params.json"
        with open(path, "w") as f:
            json.dump(best_params, f, indent=2)
        logger.info(f"Saved best params to {path}")

    return best_params


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

    # P7.3: Calibration — Platt scaling
    try:
        from sklearn.calibration import CalibratedClassifierCV
        # For models with predict_proba (LogisticRegression)
        if scaler is not None:
            X_eval_for_cal = scaler.transform(X_test)
        else:
            X_eval_for_cal = X_test

        # Platt scaling on test probabilities
        platt_probs = _platt_scale(y_prob, y_test)
        metrics["brier_score_calibrated"] = float(brier_score_loss(y_test, platt_probs))

        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_test, platt_probs, n_bins=5)
        metrics["calibration"] = {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }
        metrics["platt_params"] = {"applied": True}
    except Exception:
        metrics["calibration"] = {}
        metrics["platt_params"] = {"applied": False}

    return metrics


def _platt_scale(probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Apply Platt scaling to calibrate probabilities.

    Fits a logistic regression on log-odds of predictions to true labels.
    """
    from sklearn.linear_model import LogisticRegression as LR

    # Convert to log-odds, handle extreme values
    eps = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    # Fit Platt scaling
    platt = LR(max_iter=1000, random_state=42)
    platt.fit(log_odds, y_true)

    # Return calibrated probabilities
    calibrated_log_odds = platt.predict_log_proba(log_odds)[:, 1]
    calibrated_probs = 1.0 / (1.0 + np.exp(-calibrated_log_odds))
    return calibrated_probs


def run_training_pipeline(db: Optional[Database] = None) -> Dict[str, Any]:
    """Run the full training pipeline."""
    db = db or get_db()
    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load tuned best params (from Optuna) if available
    best_params_path = model_dir / "best_params.json"
    best_params: Dict[str, Any] = {}
    if best_params_path.exists():
        try:
            with open(best_params_path) as f:
                best_params = json.load(f)
            logger.info(f"Loaded tuned best params from {best_params_path}")
        except Exception as e:
            logger.warning(f"Could not load best params ({e}); using defaults")
            best_params = {}
    xgb_params = best_params.get("xgboost")
    lgb_params = best_params.get("lightgbm")
    if xgb_params:
        logger.info(f"Using tuned XGBoost params: {xgb_params}")
    if lgb_params:
        logger.info(f"Using tuned LightGBM params: {lgb_params}")

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
    xgb_model = train_xgboost(X_train, y_train, model_params=xgb_params)
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
            xgb_selected = train_xgboost(selected_X_train, y_train, model_params=xgb_params)
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
                num_boost_round = 300
                if xgb_params:
                    for _key in ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                                 "reg_alpha", "reg_lambda", "min_child_weight"):
                        if _key in xgb_params:
                            params[_key] = xgb_params[_key]
                    if "n_estimators" in xgb_params:
                        num_boost_round = int(xgb_params["n_estimators"])
                xgb_pseudo_model = xgb_lib.train(params, dtrain_aug, num_boost_round=num_boost_round)
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
    lgb_model = train_lightgbm(X_train, y_train, model_params=lgb_params)
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

    # P1.5: SHAP explanations
    if SHAP_AVAILABLE:
        try:
            compute_shap_values(lgb_model, X_test, feature_names, "lightgbm", model_dir)
            compute_shap_values(xgb_model, X_test, feature_names, "xgboost", model_dir)
            compute_shap_values(lr_model, X_test, feature_names, "logistic_regression", model_dir, scaler=lr_scaler)
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

    logger.info(f"Models and metrics saved to {model_dir}")
    return all_metrics


def compute_shap_values(model: Any, X: np.ndarray, feature_names: List[str], model_name: str, model_dir: Path, scaler: Any = None) -> None:
    """Compute and save SHAP values for a trained model."""
    if not SHAP_AVAILABLE:
        return
    
    logger.info(f"Computing SHAP values for {model_name}...")
    
    try:
        if model_name in ("xgboost", "lightgbm"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class for binary
        else:
            # Linear model
            if scaler is not None:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X
            explainer = shap.LinearExplainer(model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        
        # Save SHAP values
        np.savez_compressed(model_dir / f"shap_values_{model_name}.npz", 
                           shap_values=shap_values, X=X)
        
        # Compute mean absolute SHAP for feature importance
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_importance = dict(zip(feature_names, mean_abs_shap.tolist()))
        shap_importance_sorted = dict(sorted(shap_importance.items(), key=lambda x: x[1], reverse=True))
        
        with open(model_dir / f"shap_importance_{model_name}.json", "w") as f:
            json.dump(shap_importance_sorted, f, indent=2)
        
        logger.info(f"SHAP values saved for {model_name}")
        
    except Exception as e:
        logger.warning(f"SHAP computation failed for {model_name}: {e}")


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
