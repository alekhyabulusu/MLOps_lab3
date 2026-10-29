import pytest

from fastapi.testclient import TestClient
from sklearn.datasets import load_wine

from src.main import app


@pytest.fixture
def client():
    """
    Create a test client that runs FastAPI's lifespan
    events, including loading the trained model.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_wine_features():
    """
    Create a valid prediction request using the
    first sample from the Wine dataset.
    """
    wine = load_wine()

    return {
        name: float(value)
        for name, value in zip(
            wine.feature_names,
            wine.data[0]
        )
    }


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Welcome to the Wine Classification API"
    )


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "DecisionTreeClassifier"
    assert 0 <= data["accuracy"] <= 1
    assert len(data["feature_names"]) == 13
    assert len(data["target_names"]) == 3


def test_valid_prediction(client, valid_wine_features):
    response = client.post(
        "/predict",
        json={"features": valid_wine_features}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"

    prediction = data["prediction"]

    assert prediction["predicted_class"] in [0, 1, 2]
    assert prediction["class_name"] in [
        "class_0",
        "class_1",
        "class_2"
    ]

    assert 0 <= prediction["confidence"] <= 1

    probabilities = prediction["class_probabilities"]

    assert len(probabilities) == 3
    assert sum(probabilities.values()) == pytest.approx(
        1.0,
        abs=0.001
    )


def test_missing_feature(client, valid_wine_features):
    invalid_features = valid_wine_features.copy()

    invalid_features.pop("alcohol")

    response = client.post(
        "/predict",
        json={"features": invalid_features}
    )

    assert response.status_code == 422


def test_extra_feature(client, valid_wine_features):
    invalid_features = valid_wine_features.copy()

    invalid_features["unknown_feature"] = 10.0

    response = client.post(
        "/predict",
        json={"features": invalid_features}
    )

    assert response.status_code == 422