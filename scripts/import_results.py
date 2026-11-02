"""Validate and copy a completed Colab run into the static dashboard.

Usage: python scripts/import_results.py /path/to/germ_path
Only the explicit small-artifact allowlist is copied. Source files are unchanged.
No image datasets, caches, credentials, or local absolute paths are published.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
WEB_FILES = ["predictions.csv", "model_metrics.csv", "run_metadata.json", "class_lookup.csv",
             "confusion_matrix.csv", "confusion_matrix.png", "DATA_ATTRIBUTION.txt"]
ARTIFACT_FILES = ["classifier.joblib", "class_mapping.json", "frozen_protocol.json", "grouping_audit.json",
                  "resolved_versions.txt", "train_manifest.csv", "evaluation_manifest.csv"]

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def validate(source):
    metadata = json.loads((source / "run_metadata.json").read_text())
    assert metadata["protocol"]["class_codes"] == CLASSES
    assert metadata["dataset_license"] == "CC BY 4.0"
    for filename, key in [("predictions.csv", "predictions_sha256"),
                          ("train_manifest.csv", "training_manifest_sha256"),
                          ("evaluation_manifest.csv", "evaluation_manifest_sha256")]:
        assert sha256(source / filename) == metadata[key], f"Checksum mismatch: {filename}"
    with (source / "predictions.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    with (source / "evaluation_manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        manifest = {row["image_id"]: row["class_code"] for row in csv.DictReader(handle)}
    assert len(rows) == metadata["evaluated_images"] == len(manifest)
    assert len({row["image_id"] for row in rows}) == len(rows)
    assert {row["image_id"] for row in rows} == set(manifest)
    order = metadata["classifier_probability_order"]
    assert len(order) == 9 and set(order) == set(CLASSES)
    for row in rows:
        assert row["true_class"] == manifest[row["image_id"]]
        p = [float(row[f"p_{code}"]) for code in order]
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in p)
        assert abs(sum(p) - 1) < 1e-8
        maximum = max(p)
        assert row["predicted_class"] == order[p.index(maximum)]
        assert abs(float(row["confidence"]) - maximum) < 1e-10
        assert int(row["is_correct"]) == (row["true_class"] == row["predicted_class"])
        assert int(row["is_tumour_true"]) == (row["true_class"] == "TUM")
        assert row["dataset_split"] == "held_out_evaluation"
        assert row["model_name"] == metadata["protocol"]["model_name"]
    return metadata

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    source = parser.parse_args().results.resolve()
    metadata = validate(source)
    for folder, names in [(ROOT / "docs/data", WEB_FILES), (ROOT / "artifacts", ARTIFACT_FILES)]:
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(source / name, folder / name)
    print(f"Imported {metadata['evaluated_images']:,} validated predictions and artifacts.")
    print(f"Protocol: {metadata['protocol_sha256']}")
    print("Run npm test to compare the browser calculations with all fixed Python benchmarks.")

if __name__ == "__main__":
    main()
