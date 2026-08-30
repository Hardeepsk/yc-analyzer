"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_db():
    """Provide a mock database for testing."""
    db = Mock()
    db.conn = Mock()
    db.conn.execute = Mock(return_value=Mock(fetchall=Mock(return_value=[]), fetchone=Mock(return_value=None), arrow=Mock(return_value=None)))
    return db


@pytest.fixture
def sample_company_data():
    """Provide sample company data for testing."""
    return {
        "id": 1,
        "slug": "test-company",
        "name": "Test Company",
        "batch": "Winter 2024",
        "status": "Active",
        "industry": "SaaS",
        "team_size": 3,
        "tags": ["AI", "B2B"],
        "all_locations": ["San Francisco, CA, USA"],
        "website": "https://test.com",
        "top_company": False,
        "nonprofit": False,
        "is_hiring": True,
    }


@pytest.fixture
def sample_features():
    """Provide sample feature vector for testing."""
    import numpy as np
    return np.random.rand(1, 121).astype(np.float32)


@pytest.fixture
def sample_labels():
    """Provide sample labels for testing."""
    import numpy as np
    return np.array([0, 1, 0, 1, 1, 0, 0, 1])


# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark tests in tests/ as unit tests by default
        if "tests/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)