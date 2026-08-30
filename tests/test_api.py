"""Tests for API module."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from yc_analyzer.api.app import app


class TestAPI:
    """Test FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /health endpoint returns expected structure."""
        with patch("yc_analyzer.api.app.get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.conn.execute.return_value.fetchone.side_effect = [
                (100,),  # companies count
                (10,),   # batches count
                (50,),   # features count
            ]
            mock_get_db.return_value = mock_db
            
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert "companies" in data
            assert "batches" in data
            assert "features_built" in data

    def test_get_company_not_found(self, client):
        """Test /api/company/{id} returns 404 for non-existent company."""
        with patch("yc_analyzer.api.app.get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.conn.execute.return_value.fetchone.return_value = None
            mock_get_db.return_value = mock_db
            
            response = client.get("/api/company/999999")
            assert response.status_code == 404

    def test_list_batches(self, client):
        """Test /api/batches returns list of batches."""
        with patch("yc_analyzer.api.app.get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.conn.execute.return_value.fetchall.return_value = [
                ("Winter 2024", 100, 0.9, 2, 5),
                ("Summer 2023", 150, 0.85, 3, 8),
            ]
            mock_get_db.return_value = mock_db
            
            response = client.get("/api/batches")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2
            assert data[0]["batch"] == "Winter 2024"

    def test_batches_leaderboard(self, client):
        """Test /api/batches/leaderboard returns leaderboard data."""
        with patch("yc_analyzer.api.app.batch_leaderboard") as mock_leaderboard:
            mock_leaderboard.return_value = [
                {"batch": "Winter 2024", "unicorn_count": 5, "exit_count": 10},
            ]
            
            response = client.get("/api/batches/leaderboard")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_industries_endpoint(self, client):
        """Test /api/industries returns industry trends."""
        with patch("yc_analyzer.api.app.industry_trends") as mock_trends:
            mock_trends.return_value = [
                {"industry": "SaaS", "company_count": 500, "exit_rate": 0.15},
            ]
            
            response = client.get("/api/industries")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_search_companies(self, client):
        """Test /api/search returns search results."""
        with patch("yc_analyzer.api.app.get_db") as mock_get_db:
            mock_db = Mock()
            mock_db.conn.execute.return_value.fetchall.return_value = [
                (1, "Test Co", "Winter 2024", "Active", "SaaS", False, 0.8),
            ]
            mock_get_db.return_value = mock_db
            
            response = client.get("/api/search?q=test")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])