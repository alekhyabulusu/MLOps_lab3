from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "wine_model.pkl"

def load_model():
    """
    Load the trained model and its metadata.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run 'python -m src.train' first."
        )

    return joblib.load(MODEL_PATH)


def predict_wine(features, artifact):
    """
    Generate a wine classification prediction using
    a previously loaded model artifact.
    """

    model = artifact["model"]
    feature_names = artifact["feature_names"]
    target_names = artifact["target_names"]

    # Validate required features
    missing_features = [
        feature for feature in feature_names
        if feature not in features
    ]

    extra_features = [
        feature for feature in features
        if feature not in feature_names
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features: {extra_features}"
        )

    # Preserving feature order
    input_data = pd.DataFrame(
        [[features[name] for name in feature_names]],
        columns=feature_names
    )

    # Predicting class and probabilities
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    class_probabilities = {
        str(target_names[int(class_id)]): round(float(prob), 4)
        for class_id, prob in zip(model.classes_, probabilities)
    }

    return {
        "predicted_class": prediction,
        "class_name": str(target_names[prediction]),
        "confidence": class_probabilities[
            str(target_names[prediction])
        ],
        "class_probabilities": class_probabilities
    }


if __name__ == "__main__":
    from sklearn.datasets import load_wine

    wine = load_wine()

    sample = dict(
        zip(wine.feature_names, wine.data[0].tolist())
    )

    # Load model once and pass it to prediction function
    artifact = load_model()
    result = predict_wine(sample, artifact)

    print("Prediction successful!")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Class name: {result['class_name']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Class probabilities: {result['class_probabilities']}")