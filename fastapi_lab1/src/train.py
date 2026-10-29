from pathlib import Path

import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine

from src.data import load_data


# Define model storage path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"


def train_model():
    """
    Train a Decision Tree classifier with hyperparameter tuning,
    evaluate it, and save the model with metadata.
    """

    X_train, X_test, y_train, y_test = load_data()

    # hyperparameter search space
    param_grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"]
    }

    # stratified cross-validation
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # Configuring GridSearchCV
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    print("Training Decision Tree with GridSearchCV...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # evaluating on held-out test data
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nBest Hyperparameters:")
    print(grid_search.best_params_)

    print(f"\nCross-validation Accuracy: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    #  dataset metadata
    wine = load_wine()

    model_artifact = {
        "model": best_model,
        "feature_names": list(wine.feature_names),
        "target_names": list(wine.target_names),
        "accuracy": float(accuracy),
        "best_params": grid_search.best_params_
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model_artifact, MODEL_PATH)

    print(f"\nModel saved successfully to: {MODEL_PATH}")

    return model_artifact


if __name__ == "__main__":
    train_model()