import argparse
import json
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--public_dir", required=True)
    ap.add_argument("--private_dir", required=True)
    args = ap.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    public_dir = os.path.abspath(args.public_dir)
    private_dir = os.path.abspath(args.private_dir)

    train_src = os.path.join(input_dir, "train_data", "train.csv")
    test_src = os.path.join(input_dir, "predict_data", "test.csv")
    for p in (train_src, test_src):
        if not os.path.isfile(p):
            sys.exit(f"[split] missing input file: {p}")

    train = pd.read_csv(train_src)
    test = pd.read_csv(test_src)

    if "Survived" not in train.columns:
        sys.exit("[split] train.csv missing 'Survived'")
    if "PassengerId" not in train.columns or "PassengerId" not in test.columns:
        sys.exit("[split] PassengerId column required")

    # Recover original order: train_test_split preserves input order within splits,
    # so pass the original frame as-is.
    idx = range(len(train))
    y = train["Survived"].values

    train_idx, val_idx = train_test_split(
        list(idx),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    train_split = train.iloc[train_idx].reset_index(drop=True)
    val_split_full = train.iloc[val_idx].reset_index(drop=True)

    # Validation inputs: remove Survived (and no labels anywhere public)
    val_inputs = val_split_full.drop(columns=["Survived"]).reset_index(drop=True)
    # Labels only in private dir, keyed by PassengerId
    val_labels = val_split_full[["PassengerId", "Survived"]].reset_index(drop=True)

    # Ensure unique & consistent PassengerIds across public/private validation
    assert val_inputs["PassengerId"].is_unique
    assert set(val_inputs["PassengerId"]) == set(val_labels["PassengerId"])
    # Ensure no label leakage in training split duplicates: train and val disjoint
    assert set(train_split["PassengerId"]).isdisjoint(set(val_inputs["PassengerId"]))

    # Layout
    public_train = os.path.join(public_dir, "train_data")
    public_val = os.path.join(public_dir, "validation_data")
    public_test = os.path.join(public_dir, "predict_data")
    for d in (public_train, public_val, public_test, private_dir):
        os.makedirs(d, exist_ok=True)

    # Write public artifacts
    train_split.to_csv(os.path.join(public_train, "train.csv"), index=False)
    val_inputs.to_csv(os.path.join(public_val, "validation.csv"), index=False)
    test.to_csv(os.path.join(public_test, "test.csv"), index=False)

    # Write private labels
    val_labels.to_csv(os.path.join(private_dir, "validation_labels.csv"), index=False)

    # Validation schema documentation (public)
    schema_doc = {
        "validation_data/validation.csv": {
            "rows": int(len(val_inputs)),
            "columns": list(val_inputs.columns),
            "id_column": "PassengerId",
            "target_column_absent": "Survived",
            "note": "Features only. Align by PassengerId with predictions.",
        },
        "train_data/train.csv": {
            "rows": int(len(train_split)),
            "columns": list(train_split.columns),
            "note": "Labeled training split (Survived included).",
        },
        "predict_data/test.csv": {
            "rows": int(len(test)),
            "columns": list(test.columns),
            "note": "Unchanged unlabeled prediction inputs.",
        },
        "split": {
            "strategy": "train_test_split(test_size=0.2, random_state=42, stratify=Survived)",
            "seed": SEED,
            "test_size": TEST_SIZE,
            "rows_train": int(len(train_split)),
            "rows_val": int(len(val_inputs)),
            "rows_test": int(len(test)),
        },
    }
    with open(os.path.join(public_dir, "validation_schema.json"), "w") as f:
        json.dump(schema_doc, f, indent=2)

    # Smoke-test sample predictions: valid IDs, required columns, constant 0
    sample = pd.DataFrame({
        "PassengerId": val_inputs["PassengerId"].values,
        "Survived": 0,
    })
    sample.to_csv(os.path.join(public_dir, "validation_sample_predictions.csv"), index=False)

    # Print layout
    print("Split strategy: train_test_split(test_size=0.2, random_state=42, stratify=Survived)")
    print(f"input_dir  = {input_dir}")
    print(f"public_dir = {public_dir}")
    print(f"private_dir= {private_dir}")
    print(f"rows_train = {len(train_split)}")
    print(f"rows_val   = {len(val_inputs)}")
    print(f"rows_test  = {len(test)}")
    print("Public layout:")
    for root, _, files in os.walk(public_dir):
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), public_dir)
            print(f"  {rel}")
    print("Private layout:")
    for root, _, files in os.walk(private_dir):
        for fn in sorted(files):
            rel = os.path.relpath(os.path.join(root, fn), private_dir)
            print(f"  {rel}")


if __name__ == "__main__":
    main()