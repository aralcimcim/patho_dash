"""Validate completed runs and publish only small static dashboard artifacts."""

import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import CLASSES
from .common import file_hash, write_json
from .data import REFERENCE, validate_manifest
from .evaluation import fixed_checks, metrics, validate_probabilities

PUBLIC_FILES = (
    "predictions.csv",
    "model_metrics.csv",
    "metrics.json",
    "run_metadata.json",
    "manifest.csv",
)


def assert_equal(actual, expected, path="metrics"):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Metric schema mismatch: {path}")
        for key in actual:
            assert_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f"Metric list mismatch: {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            assert_equal(a, b, f"{path}[{i}]")
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        if (
            expected is None
            or not np.isfinite(expected)
            or abs(actual - expected) > 1e-9
        ):
            raise ValueError(f"Metric value mismatch: {path}")
    elif actual != expected:
        raise ValueError(f"Metric value mismatch: {path}")


def validate_run(directory):
    directory = Path(directory)
    meta = json.loads((directory / "run_metadata.json").read_text())
    if meta.get("schema_version") != 1 or meta["classifier_probability_order"] != list(
        CLASSES
    ):
        raise ValueError("Unsupported schema or noncanonical probability order.")
    if not re.fullmatch(r"[a-z0-9-]+", meta["run_id"]):
        raise ValueError("Unsafe run identifier.")
    for name, key in [
        ("predictions.csv", "predictions_sha256"),
        ("metrics.json", "metrics_sha256"),
    ]:
        if file_hash(directory / name) != meta[key]:
            raise ValueError(f"Checksum mismatch: {name}")
    frame = pd.read_csv(
        directory / "predictions.csv",
        keep_default_na=False,
        float_precision="round_trip",
    )
    if not frame.image_id.is_unique or len(frame) != meta["evaluated_images"]:
        raise ValueError("Duplicate IDs or prediction count mismatch.")
    reference = (
        pd.read_csv(REFERENCE / "evaluation_manifest.csv")
        .set_index("image_id")
        .class_code.to_dict()
    )
    if dict(zip(frame.image_id, frame.true_class)) != reference:
        raise ValueError(
            "Test image IDs or true labels differ from the recovered external test manifest."
        )
    if (
        set(frame.model_name) != {meta["protocol"]["model_name"]}
        or set(frame.run_id) != {meta["run_id"]}
        or set(frame.dataset_split) != {"held_out_evaluation"}
    ):
        raise ValueError("Run identity mismatch.")
    if meta["evaluation_counts"] != frame.true_class.value_counts().to_dict():
        raise ValueError("Evaluation class counts mismatch.")
    manifest = pd.read_csv(directory / "manifest.csv", keep_default_na=False)
    validate_manifest(manifest)
    development = manifest[manifest.split != "test"]
    original = pd.read_csv(REFERENCE / "train_manifest.csv")
    if len(development) != meta["train_images"] or dict(
        zip(development.image_id, development.class_code)
    ) != dict(zip(original.image_id, original.class_code)):
        raise ValueError(
            "Development images differ from the recovered training sample."
        )
    if (
        file_hash(directory / "manifest.csv")
        != meta["protocol"]["dataset"]["manifest_sha256"]
    ):
        raise ValueError("Manifest hash mismatch.")
    if (
        dict(
            zip(
                manifest.loc[manifest.split == "test", "image_id"],
                manifest.loc[manifest.split == "test", "class_code"],
            )
        )
        != reference
    ):
        raise ValueError("Export manifest test alignment mismatch.")
    y = frame.true_class.map(dict(zip(CLASSES, range(9)))).to_numpy()
    p = frame[[f"p_{c}" for c in CLASSES]].to_numpy(dtype=float)
    y, p = validate_probabilities(y, p)
    if (
        frame.predicted_class.to_numpy() != np.array(CLASSES)[p.argmax(1)]
    ).any() or not np.allclose(
        frame.confidence.astype(float), p.max(1), atol=1e-10, rtol=0
    ):
        raise ValueError("Predicted labels or confidence disagree with probabilities.")
    if (
        not frame.is_correct.isin([0, 1]).all()
        or not frame.is_tumour_true.isin([0, 1]).all()
    ):
        raise ValueError("Correctness and tumour flags must be zero or one.")
    if not np.array_equal(
        frame.is_correct.astype(int), y == p.argmax(1)
    ) or not np.array_equal(frame.is_tumour_true.astype(int), y == 8):
        raise ValueError("Invalid correctness/tumour flags.")
    result = metrics(y, p)
    assert_equal(result, json.loads((directory / "metrics.json").read_text()))
    checks = pd.read_csv(directory / "model_metrics.csv").set_index(
        ["class_code", "metric_name"]
    )["value"]
    expected = fixed_checks(y, p, result).set_index(["class_code", "metric_name"])[
        "value"
    ]
    if (
        not checks.index.is_unique
        or set(checks.index) != set(expected.index)
        or not np.allclose(
            checks.reindex(expected.index), expected, atol=1e-9, rtol=0, equal_nan=True
        )
    ):
        raise ValueError("Fixed dashboard checks differ from predictions.")
    return meta, result


def bundle(directories, destination):
    # Only successful, validated runs. No weights, images, embeddings or fitted classifiers.
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for directory in map(Path, directories):
            meta, _ = validate_run(directory)
            for name in PUBLIC_FILES:
                archive.write(directory / name, f"{meta['run_id']}/{name}")
    return Path(destination)


def publish(directories, docs="docs"):
    docs = Path(docs)
    registry_path = docs / "data" / "models.json"
    registry = json.loads(registry_path.read_text())
    validated = [(Path(d), *validate_run(d)) for d in directories]
    for directory, meta, _ in validated:
        target = docs / "data" / "runs" / meta["run_id"]
        if target.exists():
            if all(
                file_hash(directory / n) == file_hash(target / n) for n in PUBLIC_FILES
            ):
                continue
            raise FileExistsError(f"Published run is immutable: {target}")
        target.mkdir(parents=True)
        for name in PUBLIC_FILES:
            shutil.copyfile(directory / name, target / name)
        item = {
            "id": meta["run_id"],
            "model": meta["protocol"]["model_name"],
            "label": {
                "resnet18": "ResNet-18",
                "phikon-v2": "Phikon-v2",
                "uni2-h": "UNI2-h",
            }.get(meta["protocol"]["model_name"], meta["protocol"]["model_name"]),
            "status": "available",
            "path": f"runs/{meta['run_id']}/",
            "benchmark_version": meta["benchmark_version"],
            "historical": False,
        }
        registry["models"] = [m for m in registry["models"] if m["id"] != item["id"]]
        registry["models"].append(item)
        registry["models"] = [
            m
            for m in registry["models"]
            if not (m["status"] != "available" and m["model"] == item["model"])
        ]
        registry["default"] = item["id"]
    write_json(registry_path, registry)
