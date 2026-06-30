"""
Integration tests for the FastAPI REST API.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parents[1]))


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    from src.api.app import app
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    def test_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "docs" in data

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "loaded_models" in data

    def test_models_list(self, client):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert len(data["available_models"]) >= 1


class TestAnalyzeEndpoint:
    def test_analyze_basic(self, client):
        response = client.post("/analyze", json={"text": "I love this product!"})
        assert response.status_code == 200
        data = response.json()
        assert "label" in data
        assert "score" in data
        assert data["label"] in ("positive", "negative", "neutral")
        assert 0.0 <= data["score"] <= 1.0

    def test_analyze_positive_text(self, client):
        response = client.post("/analyze", json={
            "text": "Absolutely fantastic! Best product ever!",
            "model": "vader"
        })
        assert response.status_code == 200
        assert response.json()["label"] == "positive"

    def test_analyze_negative_text(self, client):
        response = client.post("/analyze", json={
            "text": "Terrible, disgusting, worst product ever. Awful!",
            "model": "vader"
        })
        assert response.status_code == 200
        assert response.json()["label"] == "negative"

    def test_analyze_default_model_is_vader(self, client):
        response = client.post("/analyze", json={"text": "Hello world!"})
        assert response.status_code == 200
        assert response.json()["model_name"] == "vader"

    def test_analyze_returns_scores_dict(self, client):
        response = client.post("/analyze", json={"text": "This is great!"})
        data = response.json()
        assert "scores" in data
        assert isinstance(data["scores"], dict)

    def test_analyze_returns_processing_time(self, client):
        response = client.post("/analyze", json={"text": "Test text"})
        assert "processing_time_ms" in response.json()

    def test_analyze_empty_text_rejected(self, client):
        response = client.post("/analyze", json={"text": ""})
        assert response.status_code == 422

    def test_analyze_invalid_model_rejected(self, client):
        response = client.post("/analyze", json={
            "text": "Hello",
            "model": "invalid_model"
        })
        assert response.status_code == 422

    def test_analyze_with_preprocess_false(self, client):
        response = client.post("/analyze", json={
            "text": "I love this!",
            "preprocess": False
        })
        assert response.status_code == 200

    def test_analyze_long_text(self, client):
        long_text = "This is a great product! " * 100
        response = client.post("/analyze", json={"text": long_text[:9999]})
        assert response.status_code == 200


class TestBatchEndpoint:
    def test_batch_basic(self, client):
        response = client.post("/analyze/batch", json={
            "texts": ["I love it!", "I hate it!", "It's okay."]
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_batch_each_result_has_fields(self, client):
        response = client.post("/analyze/batch", json={
            "texts": ["Great!", "Terrible!"]
        })
        for result in response.json()["results"]:
            assert "label" in result
            assert "score" in result
            assert "scores" in result
            assert "model_name" in result

    def test_batch_empty_list_rejected(self, client):
        response = client.post("/analyze/batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_returns_processing_time(self, client):
        response = client.post("/analyze/batch", json={
            "texts": ["Good", "Bad"]
        })
        assert "processing_time_ms" in response.json()

    def test_batch_preserves_order(self, client):
        texts = [
            "I absolutely love this amazing product!",   # positive
            "This is terrible and disgusting garbage!",  # negative
        ]
        response = client.post("/analyze/batch", json={"texts": texts, "model": "vader"})
        results = response.json()["results"]
        assert results[0]["label"] == "positive"
        assert results[1]["label"] == "negative"


class TestFileEndpoint:
    def test_upload_text_file(self, client, tmp_path):
        content = "I love this product!\nTerrible experience.\nPretty okay."
        file_path = tmp_path / "test.txt"
        file_path.write_text(content, encoding="utf-8")

        with open(file_path, "rb") as f:
            response = client.post(
                "/analyze/file",
                files={"file": ("test.txt", f, "text/plain")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["total_lines"] == 3

    def test_upload_invalid_extension(self, client, tmp_path):
        pdf_path = tmp_path / "file.pdf"
        pdf_path.write_bytes(b"fake pdf content")

        with open(pdf_path, "rb") as f:
            response = client.post(
                "/analyze/file",
                files={"file": ("file.pdf", f, "application/pdf")},
            )

        assert response.status_code == 422
