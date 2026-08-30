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
    """Get feature vector for a company."""
    db = db or get_db()

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

    return np.array(row, dtype=np.float32).reshape(1, -1)


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
