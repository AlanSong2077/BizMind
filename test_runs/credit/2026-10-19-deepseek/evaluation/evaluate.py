#!/usr/bin/env python3
"""Evaluate validation predictions against private labels using ROC-AUC."""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ID = "Id"
TARGET = "SeriousDlqin2yrs"
PROB = "Probability"


def write_report(private_dir, score, success, message):
    os.makedirs(private_dir, exist_ok=True)
    report = {"score": score, "success": success, "message": message}
    path = os.path.join(private_dir, "eval_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


def fail(private_dir, message):
    return write_report(private_dir, None, False, message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public_dir", required=True)
    parser.add_argument("--private_dir", required=True)
    parser.add_argument("--pred", required=True, help="Prediction CSV path")
    args = parser.parse_args()

    label_path = os.path.join(args.private_dir, "validation_labels.csv")
    if not os.path.isfile(label_path):
        return fail(args.private_dir, f"Missing private labels: {label_path}")
    if not os.path.isfile(args.pred):
        return fail(args.private_dir, f"Missing prediction file: {args.pred}")

    labels = pd.read_csv(label_path)
    if ID not in labels.columns or TARGET not in labels.columns:
        return fail(args.private_dir,
                    f"Labels file must contain '{ID}' and '{TARGET}'")

    try:
        pred = pd.read_csv(args.pred)
    except Exception as exc:  # noqa: BLE001
        return fail(args.private_dir, f"Could not read prediction file: {exc}")

    missing = [c for c in (ID, PROB) if c not in pred.columns]
    if missing:
        return fail(args.private_dir,
                    f"Prediction file missing columns {missing}; "
                    f"found {list(pred.columns)}")

    if len(pred) == 0:
        return fail(args.private_dir, "Prediction file is empty")

    if pred[ID].duplicated().any():
        dup = pred.loc[pred[ID].duplicated(), ID].tolist()[:5]
        return fail(args.private_dir, f"Duplicate Ids in predictions, e.g. {dup}")

    label_ids = set(labels[ID].tolist())
    pred_ids = set(pred[ID].tolist())
    missing_ids = label_ids - pred_ids
    extra_ids = pred_ids - label_ids
    if missing_ids:
        return fail(args.private_dir,
                    f"Missing {len(missing_ids)} validation Ids, "
                    f"e.g. {sorted(missing_ids)[:5]}")
    if extra_ids:
        return fail(args.private_dir,
                    f"Found {len(extra_ids)} extra Ids not in validation set, "
                    f"e.g. {sorted(extra_ids)[:5]}")

    prob = pd.to_numeric(pred[PROB], errors="coerce")
    if prob.isna().any():
        return fail(args.private_dir, "Prediction contains non-numeric Probability")
    if not np.isfinite(prob.to_numpy()).all():
        return fail(args.private_dir, "Prediction contains non-finite Probability")
    if (prob < 0).any() or (prob > 1).any():
        return fail(args.private_dir,
                    "Probability out of [0,1]; must be predict_proba class-1 prob")

    merged = labels[[ID, TARGET]].merge(
        pd.DataFrame({ID: pred[ID].values, PROB: prob.values}),
        on=ID, how="inner",
    )
    if len(merged) != len(labels):
        return fail(args.private_dir,
                    f"Row count mismatch after alignment: {len(merged)} vs {len(labels)}")

    score = roc_auc_score(merged[TARGET].to_numpy(), merged[PROB].to_numpy())
    return write_report(args.private_dir, float(score), True, "")


if __name__ == "__main__":
    main()