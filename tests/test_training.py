"""Tests for training module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from yc_analyzer.models.train import (
    train_baseline,
    train_xgboost,
    train_lightgbm,
    evaluate_model,
    _platt_scale,
    _compute_focal_weights,
    FEATURE_COLS,
)


class TestTraining:
    """Test ML training functions."""

    def test_feature_cols_not_empty(self):
        """Test that FEATURE_COLS is defined and non-empty."""
        assert len(FEATURE_COLS) > 0
        assert isinstance(FEATURE_COLS, list)

    def test_train_baseline_returns_model_and_scaler(self):
        """Test that train_baseline returns a model and scaler."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        model, scaler = train_baseline(X, y)
        assert model is not None
        assert scaler is not None
        assert hasattr(model, "predict_proba")

    def test_train_xgboost_returns_model(self):
        """Test that train_xgboost returns a model."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        model = train_xgboost(X, y)
        # May return None if xgboost not installed
        if model is not None:
            assert hasattr(model, "predict")

    def test_train_lightgbm_returns_model(self):
        """Test that train_lightgbm returns a model."""
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        model = train_lightgbm(X, y)
        # May return None if lightgbm not installed
        if model is not None:
            assert hasattr(model, "predict")

    def test_evaluate_model_returns_metrics(self):
        """Test that evaluate_model returns a metrics dict."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_scaled, y)
        
        metrics = evaluate_model(model, scaler, X, y, "TestModel", [f"f{i}" for i in range(10)])
        
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        assert "brier_score" in metrics
        assert "classification_report" in metrics
        assert "confusion_matrix" in metrics
        assert 0 <= metrics["auc_roc"] <= 1
        assert 0 <= metrics["auc_pr"] <= 1

    def test_platt_scale_returns_calibrated_probs(self):
        """Test that _platt_scale returns calibrated probabilities."""
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_true = np.array([0, 0, 1, 1, 1])
        
        calibrated = _platt_scale(probs, y_true)
        
        assert len(calibrated) == len(probs)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_compute_focal_weights_returns_weights(self):
        """Test that _compute_focal_weights returns weights array."""
        # The function expects 2D X, 1D y, and gamma
        X = np.array([[0.1], [0.2], [0.3], [0.7], [0.8]])  # 2D array
        y = np.array([0, 0, 0, 1, 1])
        
        weights = _compute_focal_weights(X, y, gamma=2.0)
        
        assert len(weights) == len(y)
        assert all(w >= 0 for w in weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])