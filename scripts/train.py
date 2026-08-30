#!/usr/bin/env python3
"""YC Analyzer - Train ML models for success prediction."""

import sys
import argparse
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yc_analyzer.models.labeling import compute_success_labels
from yc_analyzer.models.train import run_training_pipeline, run_tuning
from yc_analyzer.models.predict import store_predictions, predict_batch
from yc_analyzer.data.database import get_db


def main():
    parser = argparse.ArgumentParser(description="YC Analyzer - ML Training Pipeline")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run Optuna hyperparameter tuning for XGBoost/LightGBM (50 trials each) "
             "and save best params to models/best_params.json before training",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials per model when using --tune (default: 50)",
    )
    parser.add_argument(
        "--shap", action="store_true",
        help="Compute SHAP explanations after training (requires shap package)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("YC Analyzer - ML Training Pipeline")
    print("=" * 60)

    db = get_db()

    # Step 1: Compute success labels
    print("\n[1/4] Computing success labels...")
    updated = compute_success_labels(db)
    print(f"  Updated labels for {updated} companies")

    # Optional: Hyperparameter tuning
    if args.tune:
        print(f"\n[1.5] Running Optuna hyperparameter tuning ({args.trials} trials each)...")
        best_params = run_tuning(db, n_trials=args.trials)
        print("\n  Best hyperparameters found:")
        for model_name, mp in best_params.items():
            print(f"  {model_name}:")
            for k, v in mp.items():
                print(f"    {k}: {v}")
        print(f"  Saved to models/best_params.json")

    # Step 2: Train models
    print("\n[2/4] Training models...")
    metrics = run_training_pipeline(db)

    print("\n  Model Results:")
    for name, m in metrics.items():
        if isinstance(m, dict) and "error" not in m:
            print(f"  {name}:")
            print(f"    AUC-ROC:     {m.get('auc_roc', 'N/A'):.4f}")
            print(f"    AUC-PR:      {m.get('auc_pr', 'N/A'):.4f}")
            print(f"    Precision@10%: {m.get('precision_at_10pct', 'N/A'):.4f}")
            print(f"    Brier Score: {m.get('brier_score', 'N/A'):.4f}")

    # Step 3: Generate predictions for recent batches
    print("\n[3/4] Generating predictions for recent batches...")
    recent_batches = ["Winter 2024", "Spring 2024", "Summer 2024", "Fall 2024",
                      "Winter 2025", "Spring 2025"]
    total_stored = 0
    for batch in recent_batches:
        try:
            preds = predict_batch(batch, db)
            stored = store_predictions(preds, db)
            total_stored += stored
            if preds:
                avg_prob = sum(
                    p.get("ensemble", {}).get("success_probability", 0)
                    for p in preds if "ensemble" in p
                ) / len(preds)
                print(f"  {batch}: {len(preds)} companies, avg success prob = {avg_prob:.2%}")
        except Exception as e:
            print(f"  {batch}: {e}")

    print(f"  Stored {total_stored} predictions total")

    # Step 4: Save metrics
    print("\n[4/4] Saving metrics...")
    from yc_analyzer.config import settings
    import json
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.model_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Training complete! Model artifacts saved to models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
