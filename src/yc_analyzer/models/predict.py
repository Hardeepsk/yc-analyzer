"""YC Analyzer - Prediction engine."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from loguru import logger

from yc_analyzer.config import settings
from yc_analyzer.data.database import Database, get_db
from yc_analyzer.models.train import FEATURE_COLS


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

    # Get base features from DB
    row = db.conn.execute("""
        SELECT
            ce.years_since_batch, ce.team_size, ce.tag_count, ce.location_count,
            ce.has_website, ce.is_top_company, ce.is_nonprofit, ce.is_hiring,
            ce.batch_size, ce.batch_survival_rate, ce.batch_unicorn_count,
            ce.batch_exit_count, ce.batch_avg_team_size,
            ce.industry_company_count, ce.industry_exit_rate,
            ce.fed_funds_rate_at_batch, ce.nasdaq_return_1yr_post_batch,
            ce.ai_hype_index_at_batch
        FROM companies_enriched ce
        WHERE ce.company_id = ?
    """, [company_id]).fetchone()

    if row is None:
        return None

    # Unpack base features
    (years_since_batch, team_size, tag_count, location_count,
     has_website, is_top_company, is_nonprofit, is_hiring,
     batch_size, batch_survival_rate, batch_unicorn_count,
     batch_exit_count, batch_avg_team_size,
     industry_company_count, industry_exit_rate,
     fed_funds_rate, nasdaq_return, ai_hype) = row

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

    # Compute interaction features (P5.1)
    features = [
        # Base (18)
        years_since_batch, team_size, tag_count, location_count,
        1.0 if has_website else 0.0, 1.0 if is_top_company else 0.0,
        1.0 if is_nonprofit else 0.0, 1.0 if is_hiring else 0.0,
        batch_size, batch_survival_rate, batch_unicorn_count,
        batch_exit_count, batch_avg_team_size,
        industry_company_count, industry_exit_rate,
        fed_funds_rate, nasdaq_return, ai_hype,
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

    return np.array(features, dtype=np.float32).reshape(1, -1)


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
