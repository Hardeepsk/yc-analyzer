"""YC Analyzer - Prediction engine."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from loguru import logger

from yc_analyzer.config import settings
from yc_analyzer.data.database import Database, get_db
from yc_analyzer.models.train import FEATURE_COLS, NLP_FEATURE_COLS
from yc_analyzer.features.engineering import NLPEmbedder


_model_cache: Dict[str, Any] = {}


def _load_models() -> Dict[str, Any]:
    """Load trained models from disk."""
    global _model_cache
    if _model_cache:
        return _model_cache

    model_dir = settings.model_dir

    # XGBoost
    xgb_path = model_dir / "xgb_success_v1.json"
    if xgb_path.exists():
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(xgb_path))
            _model_cache["xgboost"] = model
            logger.info("Loaded XGBoost model")
        except Exception as e:
            logger.warning(f"Failed to load XGBoost: {e}")

    # LightGBM
    lgb_path = model_dir / "lgb_success_v1.txt"
    if lgb_path.exists():
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(lgb_path))
            _model_cache["lightgbm"] = model
            logger.info("Loaded LightGBM model")
        except Exception as e:
            logger.warning(f"Failed to load LightGBM: {e}")

    return _model_cache


def _get_features(company_id: int, db: Optional[Database] = None) -> Optional[np.ndarray]:
    """Get feature vector for a company, including computed interactions."""
    db = db or get_db()

    # Get base features from DB (must match FEATURE_COLS order in train.py)
    row = db.conn.execute("""
        SELECT
            ce.years_since_batch, ce.team_size, ce.tag_count, ce.location_count,
            ce.has_website, ce.is_top_company, ce.is_nonprofit, ce.is_hiring,
            ce.batch_size, ce.batch_survival_rate, ce.batch_unicorn_count,
            ce.batch_exit_count, ce.batch_avg_team_size,
            ce.industry_company_count, ce.industry_exit_rate,
            ce.fed_funds_rate_at_batch, ce.nasdaq_return_1yr_post_batch,
            ce.ai_hype_index_at_batch,
            ce.founder_count, ce.has_technical_founder, ce.has_repeat_founder,
            ce.founder_linkedin_count, ce.max_founder_bio_length
        FROM companies_enriched ce
        WHERE ce.company_id = ?
    """, [company_id]).fetchone()

    if row is None:
        return None

    # Unpack base features (18) + founder features (5)
    (years_since_batch, team_size, tag_count, location_count,
     has_website, is_top_company, is_nonprofit, is_hiring,
     batch_size, batch_survival_rate, batch_unicorn_count,
     batch_exit_count, batch_avg_team_size,
     industry_company_count, industry_exit_rate,
     fed_funds_rate, nasdaq_return, ai_hype,
     founder_count, has_technical_founder, has_repeat_founder,
     founder_linkedin_count, max_founder_bio_length) = row

    # Fill None values
    team_size = team_size or 0
    tag_count = tag_count or 0
    location_count = location_count or 0
    batch_size = batch_size or 0
    batch_survival_rate = batch_survival_rate or 0.0
    batch_unicorn_count = batch_unicorn_count or 0
    batch_exit_count = batch_exit_count or 0
    batch_avg_team_size = batch_avg_team_size or 0.0
    industry_company_count = industry_company_count or 0
    industry_exit_rate = industry_exit_rate or 0.0
    years_since_batch = years_since_batch or 0.0
    founder_count = founder_count or 0
    founder_linkedin_count = founder_linkedin_count or 0
    max_founder_bio_length = max_founder_bio_length or 0

    # Compute interaction features (P5.1) — packed into a single 16-wide block
    # that matches the order used in train.py's _prepare_xy (7 interactions +
    # 3 polynomials + 3 ratios + 3 binary flags).
    features = [
        # Base (18)
        years_since_batch, team_size, tag_count, location_count,
        1.0 if has_website else 0.0, 1.0 if is_top_company else 0.0,
        1.0 if is_nonprofit else 0.0, 1.0 if is_hiring else 0.0,
        batch_size, batch_survival_rate, batch_unicorn_count,
        batch_exit_count, batch_avg_team_size,
        industry_company_count, industry_exit_rate,
        fed_funds_rate, nasdaq_return, ai_hype,
        # Founder features (5) — must match FEATURE_COLS order
        float(founder_count),
        1.0 if has_technical_founder else 0.0,
        1.0 if has_repeat_founder else 0.0,
        float(founder_linkedin_count),
        float(max_founder_bio_length),
        # Interactions (7)
        team_size * industry_exit_rate,         # team_x_industry_exit
        team_size * batch_survival_rate,        # team_x_batch_survival
        batch_survival_rate * years_since_batch, # batch_survival_x_maturity
        industry_company_count * industry_exit_rate, # industry_density_x_exit_rate
        tag_count * team_size,                  # tags_x_team
        location_count * industry_exit_rate,    # location_x_industry_exit
        batch_unicorn_count * years_since_batch, # unicorn_density_x_maturity
        # Polynomials (3)
        team_size ** 2,                         # team_size_sq
        years_since_batch ** 2,                 # years_since_batch_sq
        batch_survival_rate ** 2,               # batch_survival_sq
        # Ratios (3)
        team_size / batch_size if batch_size > 0 else 0.0,  # team_dominance_ratio
        batch_unicorn_count / batch_size if batch_size > 0 else 0.0, # batch_unicorn_density
        batch_exit_count / batch_size if batch_size > 0 else 0.0,    # batch_exit_density
        # Binary flags (3)
        1.0 if (team_size > 10 and industry_exit_rate > 0.1) else 0.0,  # large_team_hot_industry
        1.0 if (team_size <= 5 and batch_survival_rate > 0.5) else 0.0, # small_team_strong_batch
        1.0 if (tag_count > 3 and batch_size > 100) else 0.0,           # diverse_tags_large_batch
    ]

    # P5.4: Tag rarity features
    tag_feats = _compute_single_tag_features(company_id, db)
    features.extend(tag_feats)

    # P5.3: Geographic density features
    geo_feats = _compute_single_geo_features(company_id, db)
    features.extend(geo_feats)

    # P5.5: Batch momentum features
    momentum_feats = _compute_single_momentum_features(company_id, db)
    features.extend(momentum_feats)

    # P8: NLP embedding features (loaded fitted embedder)
    nlp_feats = _compute_single_nlp_features(company_id, db)
    features.extend(nlp_feats)

    return np.array(features, dtype=np.float32).reshape(1, -1)


def _compute_single_nlp_features(company_id: int, db: Database) -> list:
    """Compute NLP embedding features for a single company (P8).

    Loads the fitted ``NLPEmbedder`` persisted during training and transforms the
    company's text fields with the same PCA/vectorizer used at training time.
    Returns a list of ``len(NLP_FEATURE_COLS)`` floats (zeros if no model is found).
    """
    embedder = NLPEmbedder.load()
    if embedder is None:
        logger.warning("No fitted NLP model found; returning zero NLP features")
        return [0.0] * len(NLP_FEATURE_COLS)
    try:
        feats = embedder.transform([company_id], db)
        return feats.astype(float).ravel().tolist()
    except Exception as e:
        logger.warning(f"NLP feature computation failed for company {company_id}: {e}")
        return [0.0] * len(NLP_FEATURE_COLS)


def _compute_single_geo_features(company_id: int, db: Database) -> list:
    """Compute geographic density features for a single company (P5.3)."""
    # Get company's locations
    row = db.conn.execute("SELECT all_locations FROM companies WHERE id = ?", [company_id]).fetchone()
    if not row or not row[0]:
        return [0.0, 0.0, 0.0]

    locations = row[0]
    if not locations:
        return [0.0, 0.0, 0.0]

    primary_loc = locations[0] if isinstance(locations, list) else locations

    # Get location stats (use CTE form — DuckDB rejects UNNEST in SELECT+GROUP BY)
    stats = db.conn.execute("""
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

    loc_unicorn = {r[0]: r[1] for r in stats}
    loc_total = {r[0]: r[2] for r in stats}

    # Top-10 hubs
    top_hubs = sorted(loc_unicorn.items(), key=lambda x: x[1], reverse=True)[:10]
    hub_set = set(h for h, _ in top_hubs)

    return [
        float(loc_unicorn.get(primary_loc, 0)),   # hub_unicorn_count
        float(loc_total.get(primary_loc, 0)),      # hub_company_count
        1.0 if primary_loc in hub_set else 0.0,    # is_in_hub
    ]


def _compute_single_momentum_features(company_id: int, db: Database) -> list:
    """Compute batch momentum features for a single company (P5.5)."""
    # Get company's batch
    row = db.conn.execute("SELECT batch FROM companies WHERE id = ?", [company_id]).fetchone()
    if not row or not row[0]:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    batch = row[0]

    # Get batch stats ordered by batch
    batch_rows = db.conn.execute("""
        SELECT batch, company_count, survival_rate, unicorn_count, exit_count
        FROM batches
        ORDER BY batch
    """).fetchall()

    if len(batch_rows) < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # Find current batch index
    batch_list = [r[0] for r in batch_rows]
    try:
        idx = batch_list.index(batch)
    except ValueError:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    if idx == 0:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    # Compute momentum vs previous batch
    curr = batch_rows[idx]
    prev = batch_rows[idx-1]

    prev_companies = prev[1] if prev[1] > 0 else 1
    curr_companies = curr[1] if curr[1] > 0 else 1

    unicorn_growth = (curr[3] / curr_companies) - (prev[3] / prev_companies)
    exit_growth = (curr[4] / curr_companies) - (prev[4] / prev_companies)
    size_growth = (curr_companies - prev_companies) / prev_companies

    # Survival trend (3-batch)
    survival_trend = 0.0
    if idx >= 3:
        survival_trend = (curr[2] - batch_rows[idx-3][2]) / 3.0

    # Composite momentum
    momentum = unicorn_growth * 2 + exit_growth * 1.5 + size_growth * 0.5 + survival_trend * 1.0

    return [unicorn_growth, exit_growth, size_growth, survival_trend, momentum]


def _compute_single_tag_features(company_id: int, db: Database) -> list:
    """Compute tag IDF, uniqueness, and trending score for a single company (P5.4)."""
    # Get company's tags
    row = db.conn.execute("SELECT tags FROM companies WHERE id = ?", [company_id]).fetchone()
    if not row or not row[0]:
        return [0.0, 0.0, 1.0]

    company_tag_list = list(set(row[0]))

    # Get all tags for IDF computation
    tag_rows = db.conn.execute("""
        SELECT id, tags, batch FROM companies
        WHERE tags IS NOT NULL AND len(tags) > 0
    """).fetchall()

    from collections import Counter
    import re
    tag_doc_freq = Counter()
    recent_tags = Counter()
    older_tags = Counter()

    for cid, tag_list, batch in tag_rows:
        if tag_list is None:
            continue
        unique_tags = list(set(tag_list))
        for t in unique_tags:
            tag_doc_freq[t] += 1
        # Trending: split by batch year
        if batch:
            m = re.search(r"(\d{4})", batch)
            if m:
                year = int(m.group(1))
                for t in unique_tags:
                    if year >= 2023:
                        recent_tags[t] += 1
                    else:
                        older_tags[t] += 1

    n_companies = len(tag_rows)
    tag_idf = {t: np.log(n_companies / max(freq, 1)) for t, freq in tag_doc_freq.items()}

    # tag_trending
    tag_trending = {}
    all_tags = set(list(recent_tags.keys()) + list(older_tags.keys()))
    for t in all_tags:
        tag_trending[t] = recent_tags.get(t, 0) / max(older_tags.get(t, 1), 1)

    # Compute features
    idf_vals = [tag_idf.get(t, 0.0) for t in company_tag_list]
    median_idf = np.median(list(tag_idf.values())) if tag_idf else 0.0
    rare_count = sum(1 for v in idf_vals if v > median_idf)
    trending_vals = [tag_trending.get(t, 1.0) for t in company_tag_list]

    return [
        float(np.mean(idf_vals)) if idf_vals else 0.0,     # tag_idf_score
        rare_count / len(company_tag_list) if company_tag_list else 0.0,  # tag_uniqueness
        float(np.mean(trending_vals)) if trending_vals else 1.0,         # tag_trending_score
    ]


def predict_company(company_id: int, db: Optional[Database] = None) -> Dict[str, Any]:
    """Predict success probability for a single company."""
    db = db or get_db()
    models = _load_models()

    if not models:
        return {"error": "No models loaded. Run training first."}

    features = _get_features(company_id, db)
    if features is None:
        return {"error": f"Company {company_id} not found or no features available"}

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Get company info
    info = db.conn.execute("""
        SELECT name, batch, status, industry, top_company
        FROM companies WHERE id = ?
    """, [company_id]).fetchone()

    result = {
        "company_id": company_id,
        "name": info[0] if info else "Unknown",
        "batch": info[1] if info else "Unknown",
        "status": info[2] if info else "Unknown",
        "industry": info[3] if info else "Unknown",
        "is_top_company": info[4] if info else False,
        "predictions": {},
    }

    # XGBoost
    if "xgboost" in models:
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix(features)
            prob = float(models["xgboost"].predict(dmat)[0])
            result["predictions"]["xgboost"] = {
                "success_probability": round(prob, 4),
                "confidence": "high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.15 else "low",
            }
        except Exception as e:
            result["predictions"]["xgboost"] = {"error": str(e)}

    # LightGBM
    if "lightgbm" in models:
        try:
            prob = float(models["lightgbm"].predict(features)[0])
            result["predictions"]["lightgbm"] = {
                "success_probability": round(prob, 4),
                "confidence": "high" if abs(prob - 0.5) > 0.3 else "medium" if abs(prob - 0.5) > 0.15 else "low",
            }
        except Exception as e:
            result["predictions"]["lightgbm"] = {"error": str(e)}

    # Ensemble average
    probs = []
    for pred in result["predictions"].values():
        if "success_probability" in pred:
            probs.append(pred["success_probability"])

    if probs:
        ensemble_prob = float(np.mean(probs))
        result["ensemble"] = {
            "success_probability": round(ensemble_prob, 4),
            "tier": (
                "high" if ensemble_prob >= 0.7 else
                "above_average" if ensemble_prob >= 0.55 else
                "average" if ensemble_prob >= 0.4 else
                "below_average" if ensemble_prob >= 0.25 else
                "low"
            ),
        }

    return result


def predict_batch(batch: str, db: Optional[Database] = None) -> List[Dict[str, Any]]:
    """Predict success for all companies in a batch."""
    db = db or get_db()

    company_ids = db.conn.execute("""
        SELECT id FROM companies WHERE batch = ?
    """, [batch]).fetchall()

    results = []
    for (cid,) in company_ids:
        pred = predict_company(cid, db)
        results.append(pred)

    # Sort by ensemble probability descending
    results.sort(
        key=lambda x: x.get("ensemble", {}).get("success_probability", 0),
        reverse=True
    )

    return results


def store_predictions(predictions: List[Dict[str, Any]], db: Optional[Database] = None) -> int:
    """Store predictions back to companies_enriched table."""
    db = db or get_db()

    # Ensure column exists
    try:
        db.conn.execute("ALTER TABLE companies_enriched ADD COLUMN IF NOT EXISTS success_at_5yr_pred DOUBLE")
    except Exception:
        pass

    count = 0
    for pred in predictions:
        if "ensemble" in pred and "success_probability" in pred["ensemble"]:
            db.conn.execute("""
                UPDATE companies_enriched
                SET success_at_5yr_pred = ?
                WHERE company_id = ?
            """, [pred["ensemble"]["success_probability"], pred["company_id"]])
            count += 1

    db.conn.commit()
    logger.info(f"Stored {count} predictions")
    return count


if __name__ == "__main__":
    # Quick test
    results = predict_batch("Winter 2024")
    print(f"Predicted {len(results)} companies in Winter 2024")
    for r in results[:5]:
        name = r.get("name", "?")
        prob = r.get("ensemble", {}).get("success_probability", "N/A")
        print(f"  {name}: {prob}")
