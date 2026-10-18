#!/usr/bin/env python3
"""Create the public/private split for the Give Me Some Credit task.

Layout produced under --public_dir:
    train_data/train.csv                            120000 labeled training rows
    validation_data/validation_features.csv         30000 unlabeled validation rows
    validation_sample_predictions.csv               smoke-test predictions (root)
    predict_data/test.csv                           unchanged 101503-row prediction set
    split_info.json                                 schema + layout documentation

Under --private_dir:
    validation_labels.csv                           Id,SeriousDlqin2yrs (labels only)

Split: single stratified 80/20 split on original row order,
test_size=0.2, random_state=42, stratify on SeriousDlqin2yrs.
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "SeriousDlqin2yrs"
ID = "Id"
RANDOM_STATE = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory containing train_data/ and predict_data/")
    parser.add_argument("--public_dir", required=True)
    parser.add_argument("--private_dir", required=True)
    args = parser.parse_args()

    train_path = os.path.join(args.input_dir, "train_data", "train.csv")
    test_path = os.path.join(args.input_dir, "predict_data", "test.csv")

    if not os.path.isfile(train_path):
        raise SystemExit(f"Missing training file: {train_path}")
    if not os.path.isfile(test_path):
        raise SystemExit(f"Missing prediction file: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    if TARGET not in train.columns:
        raise SystemExit(f"Training file missing target column '{TARGET}'")
    if ID not in train.columns or ID not in test.columns:
        raise SystemExit(f"Missing id column '{ID}'")

    # Recreate the evaluator's single stratified split on original row order.
    train_rows, val_rows = train_test_split(
        train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=train[TARGET],
    )
    train_rows = train_rows.sort_index()
    val_rows = val_rows.sort_index()

    feature_cols = [c for c in test.columns if c != ID]
    val_features = val_rows[[ID] + feature_cols].reset_index(drop=True)
    val_labels = val_rows[[ID, TARGET]].reset_index(drop=True)

    pub_train_dir = os.path.join(args.public_dir, "train_data")
    pub_val_dir = os.path.join(args.public_dir, "validation_data")
    pub_pred_dir = os.path.join(args.public_dir, "predict_data")
    for d in (pub_train_dir, pub_val_dir, pub_pred_dir, args.private_dir):
        os.makedirs(d, exist_ok=True)

    # Public: training split (labels present, only training rows).
    train_out = os.path.join(pub_train_dir, "train.csv")
    train_rows.reset_index(drop=True).to_csv(train_out, index=False)

    # Public: unlabeled validation inputs.
    val_feat_out = os.path.join(pub_val_dir, "validation_features.csv")
    val_features.to_csv(val_feat_out, index=False)

    # Public: smoke-test predictions (constant 0.5 probability, valid Ids).
    # Written at the public root so the harness/evaluator can locate it directly.
    sample = pd.DataFrame({
        ID: val_features[ID].values,
        "Probability": [0.5] * len(val_features),
    })
    sample_out = os.path.join(args.public_dir, "validation_sample_predictions.csv")
    sample.to_csv(sample_out, index=False)
    # Also keep a copy inside validation_data/ for discoverability.
    sample.to_csv(
        os.path.join(pub_val_dir, "validation_sample_predictions.csv"), index=False
    )

    # Public: unchanged unlabeled prediction set (original order, original Ids).
    pred_out = os.path.join(pub_pred_dir, "test.csv")
    test.to_csv(pred_out, index=False)

    # Private: validation labels only.
    label_out = os.path.join(args.private_dir, "validation_labels.csv")
    val_labels.to_csv(label_out, index=False)

    schema = {
        "problem_type": "binary_classification",
        "target": TARGET,
        "id_column": ID,
        "feature_columns": feature_cols,
        "split_strategy": (
            "train_test_split(test_size=0.2, random_state=42, "
            f"stratify=train['{TARGET}']) on original row order"
        ),
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(val_features)),
        "prediction_rows": int(len(test)),
        "train_positive_rate": float(train_rows[TARGET].mean()),
        "files": {
            "public_train": train_out,
            "public_validation_features": val_feat_out,
            "public_validation_sample_predictions": sample_out,
            "public_predict": pred_out,
            "private_validation_labels": label_out,
        },
    }
    with open(os.path.join(args.public_dir, "split_info.json"), "w") as f:
        json.dump(schema, f, indent=2)

    print("Split strategy:", schema["split_strategy"])
    print(f"Train rows:       {schema['train_rows']}  (pos rate {schema['train_positive_rate']:.5f})")
    print(f"Validation rows:  {schema['validation_rows']}")
    print(f"Prediction rows:  {schema['prediction_rows']}")
    print("Public files:")
    print(" ", train_out)
    print(" ", val_feat_out)
    print(" ", sample_out)
    print(" ", pred_out)
    print("Private files:")
    print(" ", label_out)
    print("  ->", os.path.join(args.public_dir, "split_info.json"))


if __name__ == "__main__":
    main()