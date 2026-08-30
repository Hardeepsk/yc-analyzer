# YC Startup Analyzer

> Historical YC data, ML success predictions, and alpha signals from **6,194 companies** across **50 batches** (2009–2026).

## Quick Start

```bash
# 1. Ingest data (already done - data/yc_analyzer.duckdb has 6,194 companies)
PYTHONPATH=src python3 scripts/ingest.py

# 2. Build features
PYTHONPATH=src python3 -m yc_analyzer.features.engineering

# 3. Train ML models
PYTHONPATH=src python3 scripts/train.py

# 4. Start API server
PYTHONPATH=src python3 scripts/serve_api.py

# 5. Start Dashboard
PYTHONPATH=src python3 scripts/serve_dashboard.py
```

## Architecture

```
src/yc_analyzer/
├── config.py              # Settings and environment
├── data/
│   ├── models.py          # Pydantic data models
│   ├── fetchers.py        # YC OSS API + Cotera fetcher
│   ├── database.py        # DuckDB schema and connection
│   └── pipeline.py        # Ingestion pipeline (upsert, dedup)
├── features/
│   └── engineering.py     # Feature engineering (founder, company, batch, market)
├── models/
│   ├── labeling.py        # Success tier labeling + temporal splits
│   ├── train.py           # XGBoost, LightGBM, LogisticRegression training
│   ├── predict.py         # Single company + batch prediction
│   └── evaluate.py        # Evaluation metrics + backtesting
├── patterns/
│   └── analyzer.py        # Batch leaderboard, industry/region/timing alpha
├── api/
│   └── app.py             # FastAPI REST API
└── dashboard/
    └── app.py             # Streamlit visualization
```

## Data Sources

| Source | Records | Auth | Update Freq |
|--------|---------|------|-------------|
| YC OSS API (`yc-oss.github.io`) | 6,194 companies | None | Daily |
| Cotera YC Dataset (parquet) | ~6,194 companies | None | Static |

**Note:** Founder data is NOT available from either source. Would require scraping individual company pages or paid API (Crunchbase, PitchBook).

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Database health check |
| `GET /api/company/{id}` | Company details |
| `GET /api/predict/{id}` | ML prediction for a company |
| `GET /api/predict/batch/{batch}` | Predict all companies in a batch |
| `GET /api/batches` | List all batches |
| `GET /api/batches/leaderboard` | Batch ranking by survival rate |
| `GET /api/industries` | Industry-level trends |
| `GET /api/timing` | Seasonal and yearly timing patterns |
| `GET /api/regions` | Geographic alpha signals |
| `GET /api/alpha` | All alpha signals combined |
| `GET /api/search?q=` | Search companies by name/industry |

## ML Models

- **XGBoost** (primary) — binary classification for 5-year success
- **LightGBM** — alternative gradient boosting model
- **LogisticRegression** — baseline comparison

Success = `top_company = True` OR `status IN (Acquired, Public)`

Features: batch metadata, industry stats, team size, timing, hiring signals. Training uses temporal splits (train on batches ≤2021, test on 2022-2023).

## Dashboard Pages

1. **Dashboard** — Overview metrics, survival by batch, industry distribution, seasonal patterns
2. **Company Search** — Search by name/industry, get ML predictions
3. **Batch Analysis** — Per-batch company tables and success probability rankings
4. **Alpha Signals** — Batch leaderboard, industry scatter, region heatmap, yearly trends
5. **Model Performance** — AUC, confusion matrix, calibration curves, feature importance

## Project Plan

Full project plan at `.slim/deepwork/yc-startup-analyzer.md` tracking all 7 phases.

## License

Data from YC OSS API (public). Cotera dataset: CC BY 4.0.
