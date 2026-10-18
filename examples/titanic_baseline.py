"""Check the official local Titanic fixture and train a baseline without an LLM."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CHECKSUMS = {
    "train.csv": "7d118fef8b6ccf7f81111877bc388536f7b1e498a655e3d649d19aaa010e9f6f",
    "test.csv": "56023b9948236f3c7a1c9448fcf418b283e109ef177fa8c7e069158dd7dd52b2",
}
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]
FEATURES = NUMERIC + CATEGORICAL


def load_data(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require the unmodified official files, including disjoint passenger IDs."""
    frames = []
    for filename, expected in CHECKSUMS.items():
        path = directory / filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{path}: checksum differs from the official downloaded fixture")
        frames.append(pd.read_csv(path))
    train, test = frames
    expected_features = {"PassengerId", "Name", "Ticket", "Cabin", *FEATURES}
    if set(train.columns) != expected_features | {"Survived"} or set(test.columns) != expected_features:
        raise ValueError("Unexpected Titanic columns or a labeled prediction dataset")
    if len(train) != 891 or len(test) != 418:
        raise ValueError("Expected 891 training rows and 418 prediction rows")
    if not train["Survived"].isin([0, 1]).all():
        raise ValueError("Training labels must be binary")
    for frame in frames:
        if frame["PassengerId"].isna().any() or not frame["PassengerId"].is_unique:
            raise ValueError("PassengerId must be present and unique")
    if set(train["PassengerId"]) & set(test["PassengerId"]):
        raise ValueError("Training and prediction passengers overlap")
    return train, test


def prediction_frame(passengers: pd.DataFrame, predictions) -> pd.DataFrame:
    """Keep input order and require binary predictions before exporting."""
    result = passengers[["PassengerId"]].copy()
    result["Survived"] = predictions
    if not result["Survived"].isin([0, 1]).all():
        raise ValueError("Predictions must contain only 0 or 1")
    result["Survived"] = result["Survived"].astype(int)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/titanic")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace/titanic-baseline")
    args = parser.parse_args()
    # Never write reports or artifacts into the downloaded source directory.
    source, output = args.data_dir.resolve(), args.output_dir.resolve()
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Use separate, non-overlapping data and output directories")
    train, test = load_data(source)
    fitting, validation = train_test_split(
        train, test_size=0.2, random_state=42, stratify=train["Survived"]
    )
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    model = Pipeline([
        ("preprocessing", ColumnTransformer([
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ])),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(fitting[FEATURES], fitting["Survived"])
    valid_predictions = prediction_frame(validation, model.predict(validation[FEATURES]))
    predictions = prediction_frame(test, model.predict(test[FEATURES]))
    score = accuracy_score(validation["Survived"], valid_predictions["Survived"])
    output.mkdir(parents=True, exist_ok=True)
    valid_predictions.to_csv(output / "validation_predictions.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    joblib.dump(model, output / "model.joblib")
    report = {
        "metric": "accuracy", "direction": "maximize", "validation_score": score,
        "train_rows": len(fitting), "validation_rows": len(validation),
        "prediction_rows": len(predictions), "random_state": 42,
        "validation_passenger_ids": validation["PassengerId"].tolist(),
        "model": "LogisticRegression", "features": FEATURES,
        "scope": "Local stratified holdout only; official test labels are unavailable.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Official Titanic CSV checksums and schemas verified: {len(train)} / {len(test)} rows")
    print(f"Local validation accuracy: {score:.6f} ({len(validation)} passengers)")
    print(f"Saved model, validation predictions, 418-row predictions and report to {output}")


if __name__ == "__main__":
    main()
