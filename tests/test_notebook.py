"""Meaningful contract tests with SYNTHETIC fixtures only; no pathology training.

Extract helper definitions from the notebook source rather than duplicating their
implementation. Run with an existing compatible numpy/pandas/sklearn/torch/Pillow
environment. This test suite is for maintainers; Colab users need no local setup.
"""
import ast
import hashlib
import json
import platform
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/01_model_evaluation.py"
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

def no_progress(iterable, **kwargs):
    return iterable

@pytest.fixture
def helpers():
    tree = ast.parse(SOURCE.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    scope = dict(globals(), tqdm=no_progress)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope

@pytest.fixture
def synthetic_predictions():
    # Hand-computable 5-row fixture, explicitly NOT model outputs.
    truth = ["ADI", "ADI", "TUM", "TUM", "BACK"]
    pairs = [("ADI", .8, "TUM", .1), ("TUM", .9, "ADI", .05),
             ("TUM", .5, "ADI", .4), ("ADI", .6, "TUM", .3)]
    p = np.zeros((5, 9))
    for i, (first, first_p, second, second_p) in enumerate(pairs):
        p[i, :] = (1 - first_p - second_p) / 7
        p[i, CLASSES.index(first)] = first_p
        p[i, CLASSES.index(second)] = second_p
    p[4, CLASSES.index("BACK")] = 1.0
    pred = np.asarray(CLASSES)[p.argmax(axis=1)]
    frame = pd.DataFrame({"image_id": [f"SYNTHETIC/{i}" for i in range(5)],
        "dataset_split": "held_out_evaluation", "model_name": "SYNTHETIC_TEST_ONLY",
        "true_class": truth, "predicted_class": pred, "confidence": p.max(axis=1),
        "is_correct": (np.asarray(truth) == pred).astype(int),
        "is_tumour_true": (np.asarray(truth) == "TUM").astype(int)})
    for j, code in enumerate(CLASSES):
        frame[f"p_{code}"] = p[:, j]
    return frame

def test_notebook_valid_compilable_and_has_no_results():
    import nbformat
    notebook = nbformat.read(ROOT / "notebooks/01_model_evaluation.ipynb", as_version=4)
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert len(code) == 15
    for cell in code:
        assert cell.execution_count is None and not cell.outputs
        compile(cell.source, cell.id, "exec")
    assert not list((ROOT / "data").rglob("predictions.csv"))

def test_export_roundtrip_and_real_probability_order(helpers, synthetic_predictions, tmp_path):
    frame = synthetic_predictions
    # Export order can differ from the classifier's actual probability order.
    helpers["validate_predictions"](frame, frame.image_id.tolist(), CLASSES[::-1])
    frame.to_csv(tmp_path / "fixture.csv", index=False, float_format="%.12g")
    back = pd.read_csv(tmp_path / "fixture.csv")
    helpers["validate_predictions"](back, frame.image_id.tolist(), CLASSES)
    assert np.allclose(frame.filter(like="p_"), back.filter(like="p_"))

@pytest.mark.parametrize("bad_field,bad_value", [
    ("p_ADI", -0.1), ("p_ADI", 1.1), ("p_ADI", float("nan")),
    ("p_ADI", .7), ("confidence", .99), ("predicted_class", "TUM"),
    ("true_class", "UNKNOWN"), ("is_correct", 0), ("is_tumour_true", 1),
    ("dataset_split", "train"),
])
def test_invalid_exports_are_rejected(helpers, synthetic_predictions, bad_field, bad_value):
    frame = synthetic_predictions.copy()
    frame.loc[0, bad_field] = bad_value
    with pytest.raises(AssertionError):
        helpers["validate_predictions"](frame, synthetic_predictions.image_id.tolist(), CLASSES)

def test_duplicate_missing_and_wrong_count_ids_rejected(helpers, synthetic_predictions):
    validate = helpers["validate_predictions"]
    expected = synthetic_predictions.image_id.tolist()
    frame = synthetic_predictions.copy()
    frame.loc[0, "image_id"] = frame.loc[1, "image_id"]
    with pytest.raises(AssertionError):
        validate(frame, expected, CLASSES)
    with pytest.raises(AssertionError):
        validate(synthetic_predictions.iloc[:-1], expected, CLASSES)
    with pytest.raises(AssertionError):
        validate(synthetic_predictions, ["SYNTHETIC/WRONG"] + expected[1:], CLASSES)

def test_known_metrics_and_threshold_boundaries(helpers, synthetic_predictions):
    metrics = helpers["metric_table"](synthetic_predictions, .8, .5, .9)
    values = metrics.loc[metrics.class_code == "ALL"].set_index("metric_name").value
    expected = {"evaluated_images": 5, "correct_predictions": 3, "accuracy": .6,
        "macro_f1": 2 / 9, "tumour_tp": 1, "tumour_fp": 1, "tumour_tn": 2, "tumour_fn": 1,
        "tumour_sensitivity": .5, "tumour_specificity": 2 / 3, "tumour_precision": .5,
        "review_images": 2, "review_proportion": .4, "coverage": .6, "retained_accuracy": 2 / 3,
        "high_confidence_images": 2, "high_confidence_errors": 1, "high_confidence_error_rate": .5}
    for name, value in expected.items():
        assert values[name] == pytest.approx(value), name
    # Empty class precision/recall is undefined; fixed-label F1 is zero.
    deb = metrics.loc[metrics.class_code == "DEB"].set_index("metric_name").value
    assert pd.isna(deb.precision) and pd.isna(deb.recall) and deb.f1 == 0
    low = helpers["metric_table"](synthetic_predictions, 0, 0, .9)
    low = low.loc[low.class_code == "ALL"].set_index("metric_name").value
    assert low.coverage == 1 and low.review_images == 0
    assert low.tumour_sensitivity == 1 and low.tumour_specificity == 0
    high = helpers["metric_table"](synthetic_predictions, 1, 1, .9)
    high = high.loc[high.class_code == "ALL"].set_index("metric_name").value
    assert high.coverage == .2 and high.retained_accuracy == 1
    assert high.tumour_sensitivity == 0 and high.tumour_specificity == 1
    assert pd.isna(high.tumour_precision)
    no_retained = synthetic_predictions.iloc[:4]
    none = helpers["metric_table"](no_retained, 1, .5, 1)
    none = none.loc[none.class_code == "ALL"].set_index("metric_name").value
    assert none.coverage == 0 and pd.isna(none.retained_accuracy)
    assert none.high_confidence_images == 0 and pd.isna(none.high_confidence_error_rate)

def test_tie_order_follows_classifier_order(helpers, synthetic_predictions):
    frame = synthetic_predictions.iloc[:1].copy()
    frame.loc[:, [f"p_{c}" for c in CLASSES]] = 0.0
    frame.loc[:, ["p_ADI", "p_TUM"]] = .5
    frame.loc[:, "confidence"] = .5
    frame.loc[:, "predicted_class"] = "TUM"
    frame.loc[:, "is_correct"] = 0
    helpers["validate_predictions"](frame, frame.image_id.tolist(), CLASSES[::-1])
    with pytest.raises(AssertionError):
        helpers["validate_predictions"](frame, frame.image_id.tolist(), CLASSES)

def test_seeded_subsampling_and_archive_metadata_stop(helpers, tmp_path):
    path = tmp_path / "SYNTHETIC.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for code in CLASSES:
            for i in range(5):
                archive.writestr(f"SYNTHETIC/{code}/{i}.tif", b"synthetic bytes, not image data")
    frame, audit = helpers["inspect_archive"](path, dict.fromkeys(CLASSES, 5))
    assert audit["grouping_available"] is False
    sample = helpers["sample_training"]
    a, b, c = sample(frame, 2, 42), sample(frame.iloc[::-1], 2, 42), sample(frame, 2, 43)
    assert a.equals(b) and not a.equals(c)
    assert a.class_code.value_counts().eq(2).all() and a.image_id.is_unique
    assert len(sample(frame, 1000, 42)) == 45
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("SYNTHETIC/patient_mapping.csv", "must be inspected before splitting")
    with pytest.raises(AssertionError, match="grouping metadata"):
        helpers["inspect_archive"](path, dict.fromkeys(CLASSES, 5))

def test_pipeline_scaler_fits_training_only():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(45, 12))
    labels = np.repeat(CLASSES, 5)
    evaluation = np.full((5, 12), 1000.0)  # Would dramatically shift a leaked scaler.
    pipeline = Pipeline([("scaler", StandardScaler()),
                         ("logreg", LogisticRegression(C=1, solver="lbfgs", max_iter=2000, random_state=42))])
    pipeline.fit(train, labels)
    before = pipeline.named_steps["scaler"].mean_.copy()
    p = pipeline.predict_proba(evaluation)
    assert np.allclose(before, train.mean(axis=0))
    assert np.array_equal(before, pipeline.named_steps["scaler"].mean_)
    assert set(pipeline.classes_) == set(CLASSES) and p.shape == (5, 9)

def test_embedding_cache_reuse_invalidation_and_inference_mode(helpers, tmp_path):
    class FakeFrozenEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, images):
            assert not self.training and not torch.is_grad_enabled()
            self.calls += 1
            return images.mean(dim=(1, 2, 3)).unsqueeze(1).repeat(1, 512)

    root = tmp_path / "SYNTHETIC_IMAGES"
    root.mkdir()
    for i in range(2):
        Image.new("RGB", (224, 224), (i, i, i)).save(root / f"{i}.tif")
    manifest = pd.DataFrame({"image_id": ["SYNTHETIC/0", "SYNTHETIC/1"], "member": ["0.tif", "1.tif"]})
    encoder = FakeFrozenEncoder()
    helpers.update(IMAGE_DIR=root, CACHE_DIR=tmp_path, BATCH_SIZE=1, NUM_WORKERS=0,
        DEVICE=torch.device("cpu"), encoder=encoder, TRANSFORM=lambda im: torch.tensor(np.asarray(im).copy()).permute(2, 0, 1).float(),
        SOURCES={"train": {"md5": "SYNTHETIC_CHECKSUM"}}, WEIGHTS_SHA256="SYNTHETIC_WEIGHTS",
        VERSIONS={"torch": torch.__version__, "torchvision": "SYNTHETIC_NOT_USED", "Pillow": "TEST"})
    a, info = helpers["extract_embeddings"](manifest, "train")
    assert encoder.calls == 2 and a.shape == (2, 512)
    b, again = helpers["extract_embeddings"](manifest, "train")
    assert encoder.calls == 2 and np.array_equal(a, b) and info == again
    helpers["BATCH_SIZE"] = 2
    _, changed = helpers["extract_embeddings"](manifest, "train")
    assert encoder.calls == 3 and changed["cache_file"] != info["cache_file"]
    # Reordering must not reuse embeddings in the wrong order.
    reverse, _ = helpers["extract_embeddings"](manifest.iloc[::-1].reset_index(drop=True), "train")
    assert np.array_equal(reverse, a[::-1])

def test_static_lookup_contracts():
    folder = ROOT / "data/lookups"
    assert pd.read_csv(folder / "class_lookup.csv").class_code.tolist() == CLASSES
    assert (folder / "class_lookup.csv").read_bytes() == (folder / "predicted_classes.csv").read_bytes()
    for name in ["thresholds", "review_thresholds", "tumour_thresholds"]:
        values = pd.read_csv(folder / f"{name}.csv").threshold.to_numpy()
        assert len(values) == 101 and np.allclose(values, np.arange(101) / 100)
    bins = pd.read_csv(folder / "confidence_bins.csv")
    for probability in [0, .1, .5, .9, 1]:
        matching = (probability >= bins.lower) & ((probability < bins.upper) | ((bins.upper == 1) & (probability <= 1)))
        assert matching.sum() == 1
