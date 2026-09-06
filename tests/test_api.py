"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client."""
    # Import here to avoid startup errors when DB isn't available
    try:
        from app.main import app
        return TestClient(app)
    except Exception:
        pytest.skip("App could not be initialized (DB required)")


class TestHealthEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestConversationEndpoints:
    def test_list_conversations(self, client):
        response = client.get("/api/conversations")
        assert response.status_code == 200
        data = response.json()
        assert "conversations" in data
        assert "total" in data


class TestDatasetEndpoints:
    def test_list_datasets(self, client):
        response = client.get("/api/datasets")
        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data

    def test_get_nonexistent_dataset(self, client):
        response = client.get("/api/datasets/nonexistent-id")
        assert response.status_code == 404
