import argparse
import json
import math
import os
import sys

import pandas as pd

REQUIRED_COLUMNS = ["PassengerId", "Survived"]


def write_report(private_dir, report):
    try:
        with open(os.path.join(private_dir, "eval_report.json"), "w") as f:
            json.dump(report, f)
    except Exception:
        pass


def fail(private_dir, message):
    report = {"score": None, "success": False, "message": message}
    write_report(private_dir, report)
    print(json.dumps(report))
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public_dir", required=True)
    ap.add_argument("--private_dir", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    public_dir = os.path.abspath(args.public_dir)
    private_dir = os.path.abspath(args.private_dir)
    pred_path = os.path.abspath(args.pred)

    os.makedirs(private_dir, exist_ok=True)

    labels_path = os.path.join(private_dir, "validation_labels.csv")
    val_inputs_path = os.path.join(public_dir, "validation_data", "validation.csv")

    if not os.path.isfile(labels_path):
        fail(private_dir, f"missing private validation labels: {labels_path}")
    if not os.path.isfile(val_inputs_path):
        fail(private_dir, f"missing public validation inputs: {val_inputs_path}")
    if not os.path.isfile(pred_path):
        fail(private_dir, f"missing prediction file: {pred_path}")

    try:
        labels = pd.read_csv(labels_path)
        val_inputs = pd.read_csv(val_inputs_path)
        pred = pd.read_csv(pred_path)
    except Exception as e:
        fail(private_dir, f"failed reading inputs: {e}")

    if list(labels.columns) != REQUIRED_COLUMNS:
        fail(private_dir, f"private labels must have columns {REQUIRED_COLUMNS}, got {list(labels.columns)}")
    if list(pred.columns) != REQUIRED_COLUMNS:
        fail(private_dir, f"prediction file must have exactly columns {REQUIRED_COLUMNS}, got {list(pred.columns)}")

    if len(pred) != len(labels):
        fail(private_dir, f"prediction row count {len(pred)} != expected {len(labels)}")

    if pred["PassengerId"].isnull().any():
        fail(private_dir, "prediction contains null PassengerId")
    if pred["PassengerId"].duplicated().any():
        dups = pred.loc[pred["PassengerId"].duplicated(), "PassengerId"].tolist()
        fail(private_dir, f"duplicate PassengerId in prediction: {dups[:10]}")

    expected_ids = set(labels["PassengerId"].tolist())
    pred_ids = set(pred["PassengerId"].tolist())
    if pred_ids != expected_ids:
        missing = sorted(list(expected_ids - pred_ids))[:10]
        extra = sorted(list(pred_ids - expected_ids))[:10]
        fail(private_dir, f"PassengerId mismatch. missing={missing} extra={extra}")

    if set(val_inputs["PassengerId"].tolist()) != expected_ids:
        fail(private_dir, "public validation input IDs do not match private label IDs")

    surv = pred["Survived"]
    if surv.isnull().any():
        fail(private_dir, "prediction contains null Survived")
    try:
        vals = pd.to_numeric(surv, errors="raise").astype(float)
    except Exception as e:
        fail(private_dir, f"Survived must be numeric: {e}")

    for v in vals.tolist():
        if not math.isfinite(v):
            fail(private_dir, "prediction contains non-finite Survived values")

    if not vals.isin([0.0, 1.0]).all():
        bad = sorted(set(vals[~vals.isin([0.0, 1.0])].tolist()))[:10]
        fail(private_dir, f"Survived must be 0 or 1, got values like {bad}")

    pred_aligned = pd.DataFrame({
        "PassengerId": pred["PassengerId"].values,
        "Survived": vals.astype(int).values,
    })

    merged = labels.merge(pred_aligned, on="PassengerId", how="inner", suffixes=("_true", "_pred"))
    if len(merged) != len(labels):
        fail(private_dir, f"internal alignment failure: merged {len(merged)} vs expected {len(labels)}")

    correct = int((merged["Survived_true"] == merged["Survived_pred"]).sum())
    total = int(len(merged))
    accuracy = correct / total if total > 0 else 0.0

    report = {"score": accuracy, "success": True, "message": ""}
    write_report(private_dir, report)
    print(json.dumps(report))
    print(f"accuracy = {correct}/{total} = {accuracy:.6f}")


if __name__ == "__main__":
    main()