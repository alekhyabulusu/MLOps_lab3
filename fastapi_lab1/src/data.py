from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data():
    """
    Load the Wine dataset and split it into
    training and testing sets.
    """

    wine = load_wine()

    X = pd.DataFrame(
        wine.data,
        columns=wine.feature_names
    )

    y = pd.Series(
        wine.target,
        name="wine_class"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    print("Wine Dataset Loaded Successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")