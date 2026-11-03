"""Canonical probability alignment, metrics, and the dashboard's 54 check values."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from . import CLASSES


def safe_ratio(a, b):
    return float(a / b) if b else None


def validate_probabilities(y, probabilities):
    y, p = np.asarray(y), np.asarray(probabilities)
    if p.shape != (len(y), 9) or not len(y) or not np.isfinite(p).all():
        raise ValueError("Invalid probability array shape or non-finite values.")
    if (
        not np.isin(y, np.arange(9)).all()
        or (p < 0).any()
        or (p > 1).any()
        or not np.allclose(p.sum(1), 1, atol=1e-8, rtol=0)
    ):
        raise ValueError("Invalid labels or probability sums.")
    return y.astype(int), p.astype(float)


def metrics(y, probabilities):
    y, p = validate_probabilities(y, probabilities)
    predicted = p.argmax(1)
    cm = confusion_matrix(y, predicted, labels=np.arange(9))
    _precision, recall, f1, support = precision_recall_fscore_support(
        y, predicted, labels=np.arange(9), zero_division=0
    )
    aucs = [
        float(roc_auc_score(y == i, p[:, i])) if 0 < (y == i).sum() < len(y) else None
        for i in range(9)
    ]
    binary = y == 8
    positive = p[:, 8] >= 0.5
    tp, fp = int((binary & positive).sum()), int((~binary & positive).sum())
    tn, fn = int((~binary & ~positive).sum()), int((binary & ~positive).sum())
    curve = []
    correct, confidence = predicted == y, p.max(1)
    for cutoff in np.linspace(0, 1, 101):
        retained = confidence >= cutoff
        accuracy = safe_ratio(int((correct & retained).sum()), int(retained.sum()))
        curve.append(
            {
                "cutoff": float(cutoff),
                "coverage": float(retained.mean()),
                "retained_accuracy": accuracy,
                "retained_error": None if accuracy is None else 1 - accuracy,
            }
        )
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()),
        "macro_auroc": float(np.mean(aucs))
        if all(a is not None for a in aucs)
        else None,
        "log_loss": float(log_loss(y, p, labels=np.arange(9))),
        "brier_score": float(np.square(p - np.eye(9)[y]).sum(1).mean()),
        "brier_definition": "mean over images of sum over nine classes of (probability - one_hot_label)^2",
        "per_class": [
            {
                "class_code": c,
                "precision": safe_ratio(int(cm[i, i]), int(cm[:, i].sum())),
                "recall": safe_ratio(int(cm[i, i]), int(support[i])),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "auroc": aucs[i],
            }
            for i, c in enumerate(CLASSES)
        ],
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": [
            [safe_ratio(int(v), int(row.sum())) for v in row] for row in cm
        ],
        "tumour": {
            "threshold": 0.5,
            "positive": "TUM",
            "negative": "all other tissue classes",
            "auroc": float(roc_auc_score(binary, p[:, 8]))
            if 0 < binary.sum() < len(y)
            else None,
            "average_precision": float(average_precision_score(binary, p[:, 8]))
            if binary.any()
            else None,
            "sensitivity": safe_ratio(tp, tp + fn),
            "specificity": safe_ratio(tn, tn + fp),
            "precision": safe_ratio(tp, tp + fp),
            "f1": safe_ratio(2 * tp, 2 * tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "matrix_order": ["other tissue", "TUM"],
        },
        "review_curve": curve,
        "probabilities": "uncalibrated",
    }


def fixed_checks(y, p, result):
    y, p = validate_probabilities(y, p)
    correct, confidence = p.argmax(1) == y, p.max(1)
    retained, high = confidence >= 0.8, confidence >= 0.9
    tumour = result["tumour"]
    values = {
        "evaluated_images": len(y),
        "correct_predictions": int(correct.sum()),
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        **{
            f"tumour_{k}": tumour[k]
            for k in ("tp", "fp", "tn", "fn", "sensitivity", "specificity", "precision")
        },
        "review_images": int((~retained).sum()),
        "review_proportion": float((~retained).mean()),
        "coverage": float(retained.mean()),
        "retained_accuracy": safe_ratio(
            int((correct & retained).sum()), int(retained.sum())
        ),
        "high_confidence_images": int(high.sum()),
        "high_confidence_errors": int((high & ~correct).sum()),
        "high_confidence_error_rate": safe_ratio(
            int((high & ~correct).sum()), int(high.sum())
        ),
    }
    rows = [
        {"metric_name": k, "class_code": "ALL", "value": v} for k, v in values.items()
    ]
    for cls in result["per_class"]:
        rows.extend(
            {"metric_name": k, "class_code": cls["class_code"], "value": cls[k]}
            for k in ("precision", "recall", "f1", "support")
        )
    return pd.DataFrame(rows)
