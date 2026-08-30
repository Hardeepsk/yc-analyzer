"""YC Analyzer - Streamlit visualization dashboard."""

import json
import sys
from pathlib import Path

# Ensure src is on path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from yc_analyzer.data.database import get_db
from yc_analyzer.models.predict import predict_company, predict_batch
from yc_analyzer.patterns.analyzer import (
    batch_leaderboard, industry_trends, timing_alpha,
    region_alpha, compute_all_alpha,
)

st.set_page_config(
    page_title="YC Startup Analyzer",
    page_icon="🚀",
    layout="wide",
)

st.title("🚀 YC Startup Analyzer")
st.markdown("Historical YC data, ML success predictions, and alpha signals from **6,194 companies** across **50 batches**.")

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Company Search", "Batch Analysis", "Alpha Signals", "Model Performance"],
)

# --- Dashboard Page ---
if page == "Dashboard":
    st.header("Overview")

    db = get_db()
    total = db.conn.execute("SELECT COUNT(*) FROM companies").fetchone()[0]
    batches = db.conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    unicorns = db.conn.execute("SELECT COUNT(*) FROM companies WHERE top_company = TRUE").fetchone()[0]
    exits = db.conn.execute("SELECT COUNT(*) FROM companies WHERE status IN ('Acquired', 'Public')").fetchone()[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Companies", f"{total:,}")
    c2.metric("Batches", batches)
    c3.metric("Unicorns", f"{unicorns:,}")
    c4.metric("Exits", f"{exits:,}")

    st.divider()

    # Survival rate by batch
    st.subheader("Survival Rate by Batch")
    lb = batch_leaderboard(db)
    if lb:
        import pandas as pd
        df = pd.DataFrame(lb)
        fig = px.bar(df, x="batch", y="survival_rate", hover_data=["unicorn_count", "exit_count"],
                     labels={"survival_rate": "Survival Rate", "batch": "Batch"})
        fig.update_layout(xaxis_tickangle=45, height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Industry distribution
    st.subheader("Top Industries by Unicorn Count")
    ind = industry_trends(db)
    if ind:
        import pandas as pd
        df_ind = pd.DataFrame(ind[:15])
        fig2 = px.bar(df_ind, x="industry", y="unicorns", color="exit_rate",
                      labels={"unicorns": "Unicorns", "industry": "Industry", "exit_rate": "Exit Rate"})
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Seasonal patterns
    st.subheader("Seasonal Success Patterns")
    timing = timing_alpha(db)
    if timing.get("seasonal_patterns"):
        import pandas as pd
        df_season = pd.DataFrame(timing["seasonal_patterns"])
        fig3 = px.bar(df_season, x="season", y="avg_survival_rate",
                      color="total_unicorns",
                      labels={"avg_survival_rate": "Avg Survival Rate", "season": "Season"})
        fig3.update_layout(height=350)
        st.plotly_chart(fig3, use_container_width=True)


# --- Company Search ---
elif page == "Company Search":
    st.header("Search Companies")

    search_query = st.text_input("Search by name or industry", placeholder="e.g. Stripe, AI, Healthcare")

    if search_query and len(search_query) >= 2:
        db = get_db()
        rows = db.conn.execute("""
            SELECT c.id, c.name, c.batch, c.status, c.industry, c.top_company, c.team_size,
                   ce.success_tier, ce.success_at_5yr_pred, ce.batch_survival_rate
            FROM companies c
            LEFT JOIN companies_enriched ce ON c.id = ce.company_id
            WHERE c.name ILIKE ? OR c.industry ILIKE ?
            ORDER BY COALESCE(ce.success_at_5yr_pred, 0) DESC
            LIMIT 50
        """, [f"%{search_query}%", f"%{search_query}%"]).fetchall()

        if rows:
            import pandas as pd
            cols = ["id", "name", "batch", "status", "industry", "top_company",
                    "team_size", "success_tier", "predicted_success", "batch_survival_rate"]
            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df, use_container_width=True)

            # Predict for selected company
            selected = st.selectbox("Select company for detailed prediction", [r[1] for r in rows])
            if selected:
                company_id = [r[0] for r in rows if r[1] == selected][0]
                with st.spinner("Predicting..."):
                    pred = predict_company(company_id)

                if "error" not in pred:
                    st.json(pred)
                    if "ensemble" in pred:
                        prob = pred["ensemble"]["success_probability"]
                        st.metric("Ensemble Success Probability", f"{prob:.1%}")
                else:
                    st.warning(pred["error"])
        else:
            st.info("No companies found matching your search.")


# --- Batch Analysis ---
elif page == "Batch Analysis":
    st.header("Batch Analysis")

    db = get_db()
    batch_list = db.conn.execute("SELECT batch FROM batches ORDER BY batch DESC").fetchall()
    batches = [r[0] for r in batch_list]

    selected_batch = st.selectbox("Select batch", batches)

    if selected_batch:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Companies in {selected_batch}")
            companies = db.conn.execute("""
                SELECT c.name, c.status, c.industry, c.team_size, c.top_company,
                       ce.success_at_5yr_pred
                FROM companies c
                LEFT JOIN companies_enriched ce ON c.id = ce.company_id
                WHERE c.batch = ?
                ORDER BY COALESCE(ce.success_at_5yr_pred, 0) DESC
            """, [selected_batch]).fetchall()

            if companies:
                import pandas as pd
                df = pd.DataFrame(companies, columns=["Name", "Status", "Industry", "Team Size",
                                                       "Top Company", "Predicted Success"])
                st.dataframe(df, use_container_width=True)

        with col2:
            st.subheader(f"Batch Predictions")
            if st.button("Run Predictions"):
                with st.spinner("Predicting..."):
                    preds = predict_batch(selected_batch)

                if preds:
                    import pandas as pd
                    pred_data = []
                    for p in preds:
                        prob = p.get("ensemble", {}).get("success_probability", 0) if "ensemble" in p else 0
                        pred_data.append({
                            "name": p.get("name", "?"),
                            "probability": prob,
                            "tier": p.get("ensemble", {}).get("tier", "N/A") if "ensemble" in p else "N/A",
                        })

                    df_pred = pd.DataFrame(pred_data)
                    fig = px.bar(df_pred, x="name", y="probability", color="tier",
                                 title=f"Success Probability - {selected_batch}")
                    fig.update_layout(xaxis_tickangle=45, height=400)
                    st.plotly_chart(fig, use_container_width=True)

                    avg_prob = df_pred["probability"].mean()
                    st.metric("Average Success Probability", f"{avg_prob:.1%}")


# --- Alpha Signals ---
elif page == "Alpha Signals":
    st.header("Alpha Signals")

    with st.spinner("Computing alpha signals..."):
        alpha = compute_all_alpha()

    # Batch leaderboard
    st.subheader("Batch Leaderboard (by Survival Rate)")
    if alpha.get("batch_leaderboard"):
        import pandas as pd
        df_lb = pd.DataFrame(alpha["batch_leaderboard"][:20])
        st.dataframe(df_lb, use_container_width=True)

    # Industry trends
    st.subheader("Industry Alpha")
    if alpha.get("industry_trends"):
        import pandas as pd
        df_ind = pd.DataFrame(alpha["industry_trends"][:15])
        fig = px.scatter(df_ind, x="total_companies", y="unicorn_rate",
                        size="unicorns", hover_name="industry",
                        labels={"total_companies": "Total Companies", "unicorn_rate": "Unicorn Rate"})
        st.plotly_chart(fig, use_container_width=True)

    # Region alpha
    st.subheader("Region Alpha (Top 20)")
    if alpha.get("region_alpha"):
        import pandas as pd
        df_reg = pd.DataFrame(alpha["region_alpha"][:20])
        fig = px.bar(df_reg, x="location", y="unicorn_rate", color="total_companies",
                     labels={"location": "Location", "unicorn_rate": "Unicorn Rate"})
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Timing
    st.subheader("Yearly Trends")
    timing = alpha.get("timing_alpha", {})
    if timing.get("yearly_trends"):
        import pandas as pd
        df_year = pd.DataFrame(timing["yearly_trends"])
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=df_year["year"], y=df_year["total_unicorns"], name="Unicorns"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_year["year"], y=df_year["avg_survival_rate"], name="Survival Rate"), secondary_y=True)
        fig.update_layout(height=400, title="YC Unicorns & Survival Rate by Year")
        st.plotly_chart(fig, use_container_width=True)


# --- Model Performance ---
elif page == "Model Performance":
    st.header("Model Performance")

    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

        for model_name, m in metrics.items():
            if isinstance(m, dict) and "error" not in m:
                st.subheader(model_name)
                c1, c2, c3 = st.columns(3)
                c1.metric("AUC-ROC", f"{m.get('auc_roc', 0):.3f}")
                c2.metric("AUC-PR", f"{m.get('auc_pr', 0):.3f}")
                c3.metric("Precision@10%", f"{m.get('precision_at_10pct', 0):.3f}")

                # Confusion matrix
                cm = m.get("confusion_matrix", [])
                if cm:
                    import pandas as pd
                    df_cm = pd.DataFrame(cm, index=["Actual Neg", "Actual Pos"],
                                         columns=["Pred Neg", "Pred Pos"])
                    fig = px.imshow(df_cm, text_auto=True, color_continuous_scale="Blues",
                                    title=f"Confusion Matrix - {model_name}")
                    st.plotly_chart(fig, use_container_width=True)

                # Calibration
                cal = m.get("calibration", {})
                if cal.get("prob_true") and cal.get("prob_pred"):
                    import pandas as pd
                    df_cal = pd.DataFrame({"actual": cal["prob_true"], "predicted": cal["prob_pred"]})
                    fig = px.line(df_cal, x="predicted", y="actual", title=f"Calibration - {model_name}")
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect", line=dict(dash="dash")))
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model metrics found. Run training first: `PYTHONPATH=src python3 scripts/train.py`")

    # Feature importance
    st.subheader("Feature Importance")
    for fname, label in [("feature_importance_xgb.json", "XGBoost"), ("feature_importance_lgb.json", "LightGBM")]:
        fpath = Path("models") / fname
        if fpath.exists():
            with open(fpath) as f:
                fi = json.load(f)
            if fi:
                import pandas as pd
                df_fi = pd.DataFrame(list(fi.items())[:15], columns=["Feature", "Importance"])
                fig = px.bar(df_fi, x="Importance", y="Feature", orientation="h",
                             title=f"Top Features - {label}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
