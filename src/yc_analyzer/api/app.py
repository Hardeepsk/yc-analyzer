"""YC Analyzer - FastAPI REST API."""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from yc_analyzer.data.database import get_db
from yc_analyzer.models.predict import predict_company, predict_batch
from yc_analyzer.patterns.analyzer import (
    batch_leaderboard, industry_trends, timing_alpha,
    region_alpha, compute_all_alpha,
)

app = FastAPI(
    title="YC Startup Analyzer",
    description="Y Combinator startup success prediction and alpha signals",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response Models ---

class HealthResponse(BaseModel):
    status: str
    companies: int
    batches: int
    features_built: bool


class CompanyPrediction(BaseModel):
    company_id: int
    name: str
    batch: str
    status: str
    industry: str
    predictions: Dict[str, Any]
    ensemble: Optional[Dict[str, Any]] = None


class BatchPrediction(BaseModel):
    batch: str
    companies: List[Dict[str, Any]]
    avg_success_probability: Optional[float] = None


# --- Routes ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    db = get_db()
    companies = db.conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    batches = db.conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    features = db.conn.execute("SELECT COUNT(*) FROM companies_enriched").fetchone()[0]
    return HealthResponse(
        status="ok",
        companies=companies,
        batches=batches,
        features_built=features > 0,
    )


@app.get("/api/company/{company_id}")
def get_company(company_id: int):
    db = get_db()
    row = db.conn.execute("""
        SELECT c.*, ce.years_since_batch, ce.batch_size, ce.batch_survival_rate,
               ce.success_tier, ce.success_at_5yr, ce.success_at_5yr_pred
        FROM companies c
        LEFT JOIN companies_enriched ce ON c.id = ce.company_id
        WHERE c.id = ?
    """, [company_id]).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Company not found")

    cols = [d[0] for d in db.conn.description]
    return dict(zip(cols, row))


@app.get("/api/predict/{company_id}")
def predict(company_id: int):
    result = predict_company(company_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/predict/batch/{batch}")
def predict_batch_endpoint(batch: str):
    results = predict_batch(batch)
    probs = [
        r["ensemble"]["success_probability"]
        for r in results
        if "ensemble" in r and "success_probability" in r.get("ensemble", {})
    ]
    return {
        "batch": batch,
        "companies": results,
        "avg_success_probability": round(float(sum(probs) / len(probs)), 4) if probs else None,
        "count": len(results),
    }


@app.get("/api/batches")
def list_batches():
    db = get_db()
    rows = db.conn.execute("""
        SELECT batch, company_count, survival_rate, unicorn_count, exit_count
        FROM batches ORDER BY batch DESC
    """).fetchall()
    return [
        {"batch": r[0], "company_count": r[1], "survival_rate": r[2],
         "unicorn_count": r[3], "exit_count": r[4]}
        for r in rows
    ]


@app.get("/api/batches/leaderboard")
def batches_leaderboard():
    return batch_leaderboard()


@app.get("/api/industries")
def industries():
    return industry_trends()


@app.get("/api/timing")
def timing():
    return timing_alpha()


@app.get("/api/regions")
def regions():
    return region_alpha()


@app.get("/api/alpha")
def alpha_signals():
    return compute_all_alpha()


@app.get("/api/search")
def search_companies(q: str = Query(..., min_length=2)):
    db = get_db()
    rows = db.conn.execute("""
        SELECT c.id, c.name, c.batch, c.status, c.industry, c.top_company,
               ce.success_at_5yr_pred
        FROM companies c
        LEFT JOIN companies_enriched ce ON c.id = ce.company_id
        WHERE c.name ILIKE ? OR c.industry ILIKE ?
        ORDER BY COALESCE(ce.success_at_5yr_pred, 0) DESC
        LIMIT 50
    """, [f"%{q}%", f"%{q}%"]).fetchall()

    return [
        {"id": r[0], "name": r[1], "batch": r[2], "status": r[3],
         "industry": r[4], "top_company": r[5], "predicted_success": r[6]}
        for r in rows
    ]
