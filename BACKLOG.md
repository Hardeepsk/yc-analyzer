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

## Priority 5 — Quick-Win Feature Engineering (HIGH ROI)
> These are low-effort, high-impact feature additions that can boost AUC by 5–15 points without external data.

### P5.1 — Feature interactions and polynomial features
- **Why**: Current features are all linear. Interactions capture non-linear patterns (e.g., "large team + fintech" is different from "large team + healthcare").
- **What**: Add interaction terms: team_size × industry_exit_rate, batch_survival_rate × years_since_batch, tag_count × team_size. Add polynomial features for top-5 features.
- **Impact**: +5–10 AUC (model can learn non-linear patterns)
- **Effort**: Low (polars column operations)

### P5.2 — Industry-batch interaction features
- **Why**: Some industries perform better in certain eras (e.g., crypto in 2021, AI in 2023). Current model treats all years the same.
- **What**: One-hot encode industry × batch_year, industry × batch_season. Add "industry momentum" (unicorn rate for this industry in last 3 batches).
- **Impact**: +3–5 AUC
- **Effort**: Low

### P5.3 — Geographic density and network features
- **Why**: SF has 2,276 companies but only 1.8% unicorn rate. Smaller hubs may have higher density of success. Proximity to other unicorns matters.
- **What**: Companies in same city as ≥5 unicorns get a boost. Add "ecosystem strength" score per location (unicorns per capita in that city). Add "hub vs satellite" binary.
- **Impact**: +2–4 AUC
- **Effort**: Low

### P5.4 — Tag co-occurrence and rarity features
- **Why**: Common tags ("SaaS", "B2B") are less predictive than rare combinations ("AI + climate + Series A").
- **What**: Compute TF-IDF weight for each tag (rare tags get higher weight). Add "tag uniqueness score" (how rare is this company's tag combination vs the batch). Add "trending tags" (tags appearing more in recent batches).
- **Impact**: +3–5 AUC
- **Effort**: Low

### P5.5 — Batch momentum and cohort features
- **Why**: Some batches are accelerating (more unicorns appearing over time). "W2020 had 2 unicorns at year 3, now has 8 at year 6" signals a strong cohort.
- **What**: Batch growth rate (unicorns added per year since batch). Batch maturity score (years since batch / expected exit timeline). "Early signal" flag (batch with above-average exits in first 3 years).
- **Impact**: +2–4 AUC
- **Effort**: Low

### P5.6 — Company age and founding timing features
- **Why**: Companies founded during recessions or market crashes behave differently. Age at YC acceptance matters.
- **What**: Company age at batch (batch_year - year_founded). "Rounded" age (companies that took >2 years to apply to YC may be more mature). "Speed to YC" (months from founding to batch).
- **Impact**: +1–3 AUC
- **Effort**: Low

---

## Priority 6 — Free Data Enrichment (HIGH ROI, NO API KEYS)
> These leverage publicly available data that can be scraped or downloaded without paid APIs.

### P6.1 — Scrape YC company pages for founder + description data
- **Why**: Founder features are ALL ZERO. This is the single biggest data gap. Company descriptions contain rich signal.
- **What**: Scrape `https://www.ycombinator.com/companies/{slug}` for: founder names, titles, LinkedIn URLs, bios, company one-liners, team size history. ~6,194 pages, ~2 hours at 1 req/sec with delays.
- **Impact**: +10–15 AUC (founder features are top predictors in YC data)
- **Effort**: Medium (beautifulsoup + rate limiting)

### P6.2 — GitHub stars and activity for open-source companies
- **Why**: Open-source companies with high GitHub stars signal developer traction and community adoption.
- **What**: For companies with GitHub links, fetch star count, fork count, contributor count, commit frequency. Add "github_popularity" and "github_momentum" features.
- **Impact**: +3–5 AUC for open-source subset
- **Effort**: Medium (GitHub API, free tier)

### P6.3 — Website quality and technology stack features
- **Why**: Companies with professional websites and modern tech stacks signal competence.
- **What**: Check if website is live (HTTP status). Detect technology stack (React, Next.js, etc.) via headers or page source. Add "has_website", "website_tech_modernity" features.
- **Impact**: +1–2 AUC
- **Effort**: Low (requests + httpx)

### P6.4 — Crunchbase free dataset integration
- **Why**: Crunchbase has funding rounds, investors, and exit data for many YC companies.
- **What**: Download Crunchbase's free dataset or scrape public profiles. Add: total_funding, last_round_size, investor_count, notable_investors, funding_velocity (funding per year).
- **Impact**: +10–15 AUC (funding data is extremely predictive)
- **Effort**: Medium

### P6.5 — PitchBook / Dealroom free datasets
- **Why**: Alternative data sources with different coverage than Crunchbase.
- **What**: Search for free PitchBook or Dealroom YC datasets on Kaggle or GitHub. Merge with existing data where Crunchbase is missing.
- **Impact**: +5–8 AUC
- **Effort**: Medium (data discovery + merging)

---

## Priority 7 — Model Architecture Improvements (HIGH ROI)

### P7.1 — Feature selection and importance pruning
- **Why**: 37 features with many zeros/noise dilute signal. Removing useless features reduces overfitting.
- **What**: Run feature importance analysis (mutual information, permutation importance). Drop features with <1% importance. Use Boruta or SHAP-based selection.
- **Impact**: +2–5 AUC (less noise, better generalization)
- **Effort**: Low

### P7.2 — Cost-sensitive learning and class weighting
- **Why**: Only 91 unicorns out of 6,194 companies (1.5%). Models are biased toward predicting "not unicorn."
- **What**: Implement focal loss or dynamic class weights. Use SMOTE for minority oversampling. Adjust decision threshold based on business cost (false negative = miss a unicorn).
- **Impact**: +3–5 AUC (better recall on rare successes)
- **Effort**: Low

### P7.3 — Model calibration improvements
- **Why**: Raw model probabilities are poorly calibrated (42% predicted success ≠ 42% actual success rate).
- **What**: Apply Platt scaling or isotonic regression as post-processing. Add calibration curves to dashboard. Make probabilities interpretable: "Of companies with 40% predicted success, 38% actually succeeded."
- **Impact**: +0 AUC but much better usability
- **Effort**: Low

### P7.4 — Time-series aware cross-validation
- **Why**: Standard k-fold CV leaks future information into training. Time-series CV prevents this.
- **What**: Implement purged k-fold CV with embargo period (gap between train/test to avoid leakage). Use expanding window or sliding window splits.
- **Impact**: More honest AUC estimates, better model selection
- **Effort**: Low

### P7.5 — Pseudo-labeling (semi-supervised learning)
- **Why**: 5,295 companies (86%) have no5-year label because they're too young. But we can use their current status as weak labels.
- **What**: Use trained model to label unlabeled companies with high confidence (>80% or <20%). Add these as training data with reduced weight. Iterate.
- **Impact**: +3–5 AUC (more training data)
- **Effort**: Low

---

## Priority 8 — Advanced Features

### P8.1 — Batch success predictor (predict before joining YC)
- **Why**: Founders want to know "will my batch be good?"
- **What**: Predict batch-level success rate before demo day based on composition (industries, team sizes, geography).
- **Impact**: Unique value proposition
- **Effort**: Medium

### P8.2 — "Similar companies" finder
- **Why**: Users want to find analogues to their company.
- **What**: Embedding-based similarity search (sentence-transformers on company descriptions + features).
- **Impact**: Discovery feature
- **Effort**: Medium

### P8.3 — What-if scenario builder
- **Why**: Users want to know "if I had a technical co-founder, how would my score change?"
- **What**: Interactive tool that lets users modify features and see real-time prediction changes. Show which features would have the biggest impact.
- **Impact**: Actionable insights (not just predictions)
- **Effort**: Medium

### P8.4 — Investor quality scoring
- **Why**: Some investors have 10x better track records. Knowing "this investor backed 3 unicorns" matters.
- **What**: Build investor network graph. Score investors by portfolio success rate. Add "investor_quality" feature to model.
- **Impact**: +5–8 AUC (investor signal is strong)
- **Effort**: Medium-High

### P8.5 — Market timing dashboard
- **Why**: Macro conditions affect startup success. Users want "is now a good time to start?"
- **What**: Fed funds rate, NASDAQ returns, VC funding trends overlaid with YC batch performance.
- **Impact**: Macro-alpha signal
- **Effort**: Medium (external data sourcing)

### P8.6 — Competitive landscape analysis
- **Why**: "How many AI companies are in my batch?" matters. Crowded batches may have lower individual success.
- **What**: Compute industry density per batch, competitor count per company, market share proxy.
- **Impact**: +2–3 AUC
- **Effort**: Low

### P8.7 — Real-time YC company monitoring
- **Why**: Track new YC companies as they launch.
- **What**: Daily check for new companies, auto-predict, notify via email/Slack webhook.
- **Impact**: Early signal advantage
- **Effort**: Medium

### P8.8 — Multi-model ensemble with stacking
- **Why**: Current ensemble is simple average. Stacking can improve.
- **What**: Train a meta-learner on top of XGBoost/LightGBM/LR predictions + original features.
- **Impact**: +2–5 AUC
- **Effort**: Low

---

## Priority 9 — Product & Distribution (MEDIUM ROI)

### P9.1 — Chrome extension for YC company pages
- **Why**: Users browse YC companies daily. Show predictions inline.
- **What**: Extension that adds "YC Analyzer Score" badge to every company on ycombinator.com/companies.
- **Impact**: Viral distribution, daily usage
- **Effort**: Medium

### P9.2 — Slack bot for company lookups
- **Why**: Teams discuss YC companies in Slack. Quick lookups without leaving chat.
- **What**: `/yc-score company-name` returns prediction, batch info, similar companies.
- **Impact**: Team adoption
- **Effort**: Low

### P9.3 — Email digest for new batches
- **Why**: Users want to know when new YC batches are announced.
- **What**: Weekly email with batch analysis, top predicted companies, industry trends.
- **Impact**: Retention
- **Effort**: Medium

### P9.4 — PDF report generation
- **Why**: Users want to share analysis with partners/team.
- **What**: One-click PDF report with company profile, prediction, SHAP explanations, batch context.
- **Impact**: Professional output
- **Effort**: Medium

### P9.5 — Embeddable widgets
- **Why**: YC-focused blogs and newsletters want to embed predictions.
- **What**: `<iframe>` or JS widget that shows "YC Analyzer Score" for any company.
- **Impact**: Distribution channel
- **Effort**: Medium

---

## Priority 10 — Infrastructure & Quality

### P10.1 — Docker Compose setup
- **Why**: Currently requires manual Python setup. Docker makes deployment trivial.
- **What**: Dockerfile + docker-compose.yml with API, dashboard, and DuckDB volume.
- **Impact**: One-command deployment
- **Effort**: Low

### P10.2 — Unit tests for core modules
- **Why**: No tests exist. Regressions are invisible.
- **What**: pytest suite for labeling, training, prediction, feature engineering. Target 80% coverage.
- **Impact**: Confidence in changes
- **Effort**: Medium

### P10.3 — GitHub Actions CI/CD
- **Why**: No automated testing or deployment.
- **What**: CI pipeline: lint (ruff) → type check (mypy) → tests → build.
- **Impact**: Quality gates
- **Effort**: Low

### P10.4 — API caching with Redis
- **Why**: Repeated predictions hit the model every time. Predictions don't change often.
- **What**: Cache prediction results in Redis with 24h TTL. Invalidate on model retrain.
- **Impact**: 10x faster API responses
- **Effort**: Low

### P10.5 — Monitoring and alerting
- **Why**: No visibility into model drift or data quality issues.
- **What**: Track prediction distribution over time, data freshness, API latency. Alert on anomalies.
- **Impact**: Operational reliability
- **Effort**: Medium

---

## Backlog (Nice to Have)
- [ ] TypeScript frontend (replace Streamlit)
- [ ] GraphQL API
- [ ] Mobile app
- [ ] A/B testing framework for model versions
- [ ] Feature store (Feast)
- [ ] Model registry (MLflow)
- [ ] Monitoring dashboard (Grafana)
- [ ] Multi-tenant API with user accounts
- [ ] Natural language queries ("show me AI companies with technical founders")
- [ ] Batch comparison tool ("compare W2020 vs W2021")
- [ ] Survival curve visualization per company
- [ ] Investor portfolio overlap analysis

---

## Estimated Timeline
| Priority | Items | Effort | Impact | ROI |
|----------|-------|--------|--------|-----|
| P1 (Model Performance) | 5 items | 2–3 weeks | 🔴 Critical | High |
| P2 (Data Quality) | 4 items | 1–2 weeks | 🟠 High | High |
| P5 (Quick-Win Features) | 6 items | 1 week | 🟠 High | 🔥 Highest |
| P6 (Free Data) | 5 items | 1–2 weeks | 🟠 High | 🔥 Highest |
| P7 (Model Architecture) | 5 items | 1 week | 🟠 High | 🔥 Highest |
| P8 (Advanced) | 8 items | 2–3 weeks | 🟢 Nice | Medium |
| P3 (API/Dashboard) | 5 items | 1–2 weeks | 🟡 Medium | Medium |
| P9 (Product) | 5 items | 2 weeks | 🟡 Medium | Medium |
| P4 (Testing) | 4 items | 1 week | 🟡 Medium | Medium |
| P10 (Infrastructure) | 5 items | 1 week | 🟡 Medium | Medium |

**Recommended next sprint (max ROI):**
1. P5.1 (feature interactions) — 1 day, +5-10 AUC
2. P5.4 (tag rarity) — 1 day, +3-5 AUC
3. P6.1 (scrape founders) — 2 days, +10-15 AUC
4. P7.1 (feature selection) — 1 day, +2-5 AUC
5. P7.5 (pseudo-labeling) — 1 day, +3-5 AUC

**Total effort: 1 week. Expected AUC improvement: 0.57 → 0.70+**

---

## Quick Reference: Feature Impact Estimate
| Feature Type | Current | After | AUC Gain |
|--------------|---------|-------|----------|
| Feature interactions | None | 20+ interactions | +5–10 |
| Industry-batch timing | None | Industry×Year features | +3–5 |
| Geographic density | None | Hub strength scores | +2–4 |
| Tag rarity (TF-IDF) | Raw tags | Weighted tag scores | +3–5 |
| Batch momentum | None | Growth rate features | +2–4 |
| Founder features | All zero | Scrape from YC pages | +10–15 |
| GitHub activity | None | Stars, forks, contributors | +3–5 |
| Funding data | None | Total raised, round count | +10–15 |
| Feature selection | All 37 | Top 15-20 | +2–5 |
| Class balancing | Default | Focal loss / SMOTE | +3–5 |
| Pseudo-labeling | None | Semi-supervised | +3–5 |
| **TOTAL (estimated)** | **0.57** | **0.75+** | **+18–30** |
