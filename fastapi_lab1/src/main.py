from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator
from src.predict import load_model, predict_wine

# Load model once during application startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_artifact = load_model()
    yield

app = FastAPI(
    title="Wine Classification API",
    description="A machine learning API using FastAPI and Decision Trees.",
    version="1.0.0",
    lifespan=lifespan
)


# Request schema
class WineFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: Dict[str, float] = Field(
        ...,
        description="Dictionary containing the 13 wine features"
    )

    @model_validator(mode="after")
    def validate_features(self):
        expected_features = set(
            [
                "alcohol",
                "malic_acid",
                "ash",
                "alcalinity_of_ash",
                "magnesium",
                "total_phenols",
                "flavanoids",
                "nonflavanoid_phenols",
                "proanthocyanins",
                "color_intensity",
                "hue",
                "od280/od315_of_diluted_wines",
                "proline"
            ]
        )

        provided_features = set(self.features.keys())

        missing = expected_features - provided_features
        extra = provided_features - expected_features

        if missing:
            raise ValueError(
                f"Missing features: {sorted(missing)}"
            )

        if extra:
            raise ValueError(
                f"Unexpected features: {sorted(extra)}"
            )

        return self


# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to the Wine Classification API",
        "version": "1.0.0",
        "available_endpoints": [
            "/",
            "/health",
            "/predict",
            "/model-info",
            "/docs"
        ]
    }


# Health check endpoint
@app.get("/health")
def health_check():
    try:
        artifact = app.state.model_artifact

        return {
            "status": "healthy",
            "model_loaded": artifact["model"] is not None
        }

    except AttributeError:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )


# Model metadata endpoint
@app.get("/model-info")
def model_info():
    try:
        artifact = app.state.model_artifact

        return {
            "model_type": type(artifact["model"]).__name__,
            "accuracy": artifact["accuracy"],
            "best_parameters": artifact["best_params"],
            "feature_names": artifact["feature_names"],
            "target_names": artifact["target_names"]
        }

    except AttributeError:
        raise HTTPException(
            status_code=503,
            detail="Model information is unavailable"
        )


# Prediction endpoint
@app.post("/predict")
def predict(request: WineFeatures):
    try:
        artifact = app.state.model_artifact

        result = predict_wine(
            request.features,
            artifact
        )

        return {
            "status": "success",
            "prediction": result
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except AttributeError:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )