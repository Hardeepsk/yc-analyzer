# YC Startup Analyzer — Product Backlog

## Current State (v0.1.0)
- **6,194 companies** across 50 batches (2009–2026)
- **37 feature columns** in companies_enriched
- **ML models trained**: XGBoost (AUC 0.48), LightGBM (AUC 0.46), LogisticRegression (AUC 0.57)
- **999 predictions** stored for recent batches
- **FastAPI** with 12 endpoints, **Streamlit** dashboard with 5 pages
- **GitHub**: https://github.com/Hardeepsk/yc-analyzer

---

## Priority 1 — Model Performance (CRITICAL)
> Model AUCs are near random (0.46–0.57). These need significant improvement before the tool is useful.

### P1.1 — Enrich with Crunchbase/Signal data
- **Why**: Current features are mostly batch-level aggregates. No funding amounts, investor quality, or founder backgrounds.
- **What**: Add funding rounds, total raised, valuation, investor names, round timing from Crunchbase API or pre-built datasets (e.g. `kaggle.com/datasets/`).
- **Impact**: +20–30 AUC points expected. This is the single biggest lever.
- **Effort**: Medium (data sourcing + new features)

### P1.2 — Add text/NLP features from company descriptions
- **Why**: Company name, tags, and industry descriptions contain signal not captured in numeric features.
- **What**: TF-IDF or sentence embeddings on company descriptions, tag lists, and industry text. Use pre-trained models (sentence-transformers).
- **Impact**: +5–10 AUC
- **Effort**: Medium

### P1.3 — Survival analysis (time-to-event modeling)
- **Why**: Current labels treat success as binary. Real outcome is time-dependent (some companies take 7+ years to exit).
- **What**: Cox proportional hazards or DeepSurv. Predict "probability of success by year X" instead of just "success at 5yr."
- **Impact**: Better calibration, more actionable predictions
- **Effort**: Medium (lifelines already installed)

### P1.4 — Hyperparameter tuning with Optuna
- **Why**: Current hyperparameters are defaults. Optuna can find better configs.
- **What**: 50-trial Bayesian optimization for XGBoost/LightGBM with time-series cross-validation.
- **Impact**: +3–5 AUC
- **Effort**: Low

### P1.5 — SHAP explanations for every prediction
- **Why**: Black-box predictions aren't actionable. Users need "why is this company predicted to succeed?"
- **What**: Add SHAP waterfall plots per company, feature contribution explanations in API responses and dashboard.
- **Impact**: Usability breakthrough
- **Effort**: Low (shap already installed)

---

## Priority 2 — Data Quality & Freshness

### P2.1 — Automated daily data refresh
- **Why**: YC OSS API updates daily. Currently manual re-run.
- **What**: Cron job or GitHub Action that runs `scripts/ingest.py` daily, re-computes features, re-trains models weekly.
- **Impact**: Data stays fresh
- **Effort**: Low

### P2.2 — Scrape founder data from individual company pages
- **Why**: Founder features are all zero (biggest data gap). Repeat founders, technical founders, and school prestige are strong predictors.
- **What**: Scrape `https://www.ycombinator.com/companies/{slug}` for founder names, LinkedIn URLs, bios. ~6,194 pages, ~2 hours at 1 req/sec.
- **Impact**: +10–15 AUC (founder features are among the strongest signals)
- **Effort**: Medium-High (rate limiting, pagination)

### P2.3 — Backfill historical predictions for all companies
- **Why**: Only 999 companies have predictions. Need all 6,194 for dashboard completeness.
- **What**: Run `predict_batch()` for every batch in the database.
- **Impact**: Complete dashboard
- **Effort**: Low (30 min runtime)

### P2.4 — Add Cotera dataset enrichment
- **Why**: Cotera parquet file downloaded but not fully integrated. May contain additional fields.
- **What**: Merge Cotera columns into companies table where YC OSS data is missing.
- **Impact**: +5–10% data completeness
- **Effort**: Low

---

## Priority 3 — API & Dashboard Polish

### P3.1 — Docker Compose setup
- **Why**: Currently requires manual Python setup. Docker makes deployment trivial.
- **What**: Dockerfile + docker-compose.yml with API, dashboard, and DuckDB volume.
- **Impact**: One-command deployment
- **Effort**: Low

### P3.2 — API rate limiting and auth
- **Why**: Public API without limits is a liability.
- **What**: SlowAPI rate limiting, optional API key auth for high-volume users.
- **Impact**: Production-ready API
- **Effort**: Low

### P3.3 — Dashboard UX improvements
- **Why**: Dashboard works but lacks polish.
- **What**: Loading states, error boundaries, responsive design, dark mode, mobile-friendly layout, favicon.
- **Impact**: Better user experience
- **Effort**: Medium (@designer)

### P3.4 — Add company comparison tool
- **Why**: Users want to compare two companies side-by-side.
- **What**: New dashboard page + API endpoint for head-to-head comparison (features, predictions, batch context).
- **Impact**: High-utility feature
- **Effort**: Medium

### P3.5 — Export functionality (CSV, PDF reports)
- **Why**: Users want to download data and reports.
- **What**: API endpoints for CSV export, PDF report generation with charts and predictions.
- **Impact**: Professional-grade output
- **Effort**: Medium

---

## Priority 4 — Testing & Reliability

### P4.1 — Unit tests for core modules
- **Why**: No tests exist. Regressions are invisible.
- **What**: pytest suite for labeling, training, prediction, feature engineering. Target 80% coverage.
- **Impact**: Confidence in changes
- **Effort**: Medium

### P4.2 — Integration tests (end-to-end pipeline)
- **Why**: Need to verify full pipeline works after changes.
- **What**: Test script that runs ingest → features → labels → train → predict → API health check.
- **Impact**: Catch integration bugs
- **Effort**: Medium

### P4.3 — GitHub Actions CI/CD
- **Why**: No automated testing or deployment.
- **What**: CI pipeline: lint (ruff) → type check (mypy) → tests → build.
- **Impact**: Quality gates
- **Effort**: Low

### P4.4 — Error handling and logging improvements
- **Why**: Some modules have bare except clauses or missing error handling.
- **What**: Structured logging, retry logic for API calls, graceful degradation when models aren't trained.
- **Impact**: Robustness
- **Effort**: Low

---

## Priority 5 — Advanced Features

### P5.1 — Batch success predictor (predict before joining YC)
- **Why**: Founders want to know "will my batch be good?"
- **What**: Predict batch-level success rate before demo day based on composition (industries, team sizes, geography).
- **Impact**: Unique value proposition
- **Effort**: Medium

### P5.2 — "Similar companies" finder
- **Why**: Users want to find analogues to their company.
- **What**: Embedding-based similarity search (sentence-transformers on company descriptions + features).
- **Impact**: Discovery feature
- **Effort**: Medium

### P5.3 — Market timing dashboard
- **Why**: Macro conditions affect startup success. Users want "is now a good time to start?"
- **What**: Fed funds rate, NASDAQ returns, VC funding trends overlaid with YC batch performance.
- **Impact**: Macro-alpha signal
- **Effort**: Medium (external data sourcing)

### P5.4 — Real-time YC company monitoring
- **Why**: Track new YC companies as they launch.
- **What**: Daily check for new companies, auto-predict, notify via email/Slack webhook.
- **Impact**: Early signal advantage
- **Effort**: Medium

### P5.5 — Multi-model ensemble with stacking
- **Why**: Current ensemble is simple average. Stacking can improve.
- **What**: Train a meta-learner on top of XGBoost/LightGBM/LR predictions + original features.
- **Impact**: +2–5 AUC
- **Effort**: Low

---

## Backlog (Nice to Have)
- [ ] TypeScript frontend (replace Streamlit)
- [ ] GraphQL API
- [ ] Mobile app
- [ ] Chrome extension for YC company pages
- [ ] Slack bot for company lookups
- [ ] A/B testing framework for model versions
- [ ] Feature store (Feast)
- [ ] Model registry (MLflow)
- [ ] Monitoring dashboard (Grafana)
- [ ] Multi-tenant API with user accounts

---

## Estimated Timeline
| Priority | Items | Effort | Impact |
|----------|-------|--------|--------|
| P1 (Model Performance) | 5 items | 2–3 weeks | 🔴 Critical |
| P2 (Data Quality) | 4 items | 1–2 weeks | 🟠 High |
| P3 (API/Dashboard) | 5 items | 1–2 weeks | 🟡 Medium |
| P4 (Testing) | 4 items | 1 week | 🟡 Medium |
| P5 (Advanced) | 5 items | 2–3 weeks | 🟢 Nice |

**Recommended next sprint:** P1.1 (Crunchbase enrichment) + P1.5 (SHAP) + P2.3 (backfill predictions) + P3.1 (Docker) + P4.1 (tests)
