"""CPU checks only: synthetic predictions are fixtures, never benchmark results."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import CLASSES, VERSION
from benchmark.common import file_hash, write_json
from benchmark.data import assign_splits, read_reference, validate_manifest
from benchmark.evaluation import fixed_checks, metrics, validate_probabilities
from benchmark.publish import bundle, publish, validate_run
from benchmark.run import select_classifier

ROOT = Path(__file__).resolve().parents[1]


def fixture(directory):
    manifest = assign_splits(read_reference())
    manifest["sha256"] = [f"{i:064x}" for i in range(len(manifest))]
    test = manifest[manifest.split == "test"]
    y = test.class_index.to_numpy()
    p = np.full((len(y), 9), 0.025)
    p[np.arange(len(y)), y] = 0.8
    predicted = p.argmax(1)
    frame = pd.DataFrame(
        {
            "image_id": test.image_id.to_numpy(),
            "dataset_split": "held_out_evaluation",
            "model_name": "resnet18",
            "run_id": "resnet18-test-fixture",
            "true_class": test.class_code.to_numpy(),
            "predicted_class": np.array(CLASSES)[predicted],
            "confidence": p.max(1),
            "is_correct": (predicted == y).astype(int),
            "is_tumour_true": (y == 8).astype(int),
        }
    )
    for i, code in enumerate(CLASSES):
        frame[f"p_{code}"] = p[:, i]
    directory.mkdir(parents=True)
    frame.to_csv(directory / "predictions.csv", index=False, float_format="%.17g")
    manifest.to_csv(directory / "manifest.csv", index=False)
    result = metrics(y, p)
    write_json(directory / "metrics.json", result)
    fixed_checks(y, p, result).to_csv(
        directory / "model_metrics.csv", index=False, float_format="%.17g"
    )
    metadata = {
        "schema_version": 1,
        "benchmark_version": VERSION,
        "run_id": "resnet18-test-fixture",
        "classifier_probability_order": CLASSES,
        "evaluated_images": len(y),
        "train_images": 9000,
        "evaluation_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)},
        "completed_at_utc": "2026-01-01T00:00:00Z",
        "protocol": {
            "model_name": "resnet18",
            "dataset": {"manifest_sha256": file_hash(directory / "manifest.csv")},
            "review_cutoff": 0.8,
            "tumour_threshold": 0.5,
        },
        "predictions_sha256": file_hash(directory / "predictions.csv"),
        "metrics_sha256": file_hash(directory / "metrics.json"),
    }
    write_json(directory / "run_metadata.json", metadata)
    return metadata, result


class BenchmarkTests(unittest.TestCase):
    def test_original_sample_shared_split(self):
        first, second = assign_splits(read_reference()), assign_splits(read_reference())
        pd.testing.assert_frame_equal(first, second)
        validate_manifest(first)
        self.assertEqual(
            first.split.value_counts().to_dict(),
            {"train": 7200, "test": 7180, "validation": 1800},
        )
        self.assertEqual(
            first[first.split != "test"].class_code.value_counts().to_dict(),
            dict.fromkeys(CLASSES, 1000),
        )
        different = assign_splits(read_reference(), seed=17)
        self.assertEqual(
            set(first[first.split == "test"].image_id),
            set(different[different.split == "test"].image_id),
        )
        self.assertNotEqual(
            set(first[first.split == "validation"].image_id),
            set(different[different.split == "validation"].image_id),
        )
        self.assertTrue((first.patient_id == "").all())

    def test_manifest_rejects_leakage_and_invalid_mapping(self):
        original = assign_splits(read_reference())
        for column, value in [
            ("image_id", original.image_id.iloc[1]),
            ("class_index", 100),
            ("relative_path", "../escape"),
        ]:
            frame = original.copy()
            frame.loc[0, column] = value
            with self.assertRaises(ValueError):
                validate_manifest(frame)
        frame = original.copy()
        frame["sha256"] = [f"{i:064x}" for i in range(len(frame))]
        first = frame.index[frame.split == "train"][0]
        other = frame.index[frame.split == "test"][0]
        frame.loc[first, "sha256"] = frame.loc[other, "sha256"]
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            validate_manifest(frame)

    def test_metrics_brier_and_argmax_are_separate(self):
        p = np.eye(9)
        result = metrics(np.arange(9), p)
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["brier_score"], 0)
        self.assertEqual(result["tumour"]["confusion_matrix"], [[8, 0], [0, 1]])
        wrong = np.roll(p, 1, axis=1)
        self.assertEqual(metrics(np.arange(9), wrong)["brier_score"], 2)
        probabilities = np.tile(np.array([0.09] * 8 + [0.28]), (9, 1))
        result = metrics(np.arange(9), probabilities)
        self.assertEqual(result["confusion_matrix"][8][8], 1)  # Argmax TUM correct.
        self.assertEqual(result["tumour"]["sensitivity"], 0)  # p_TUM < 0.5.
        self.assertIsNone(result["review_curve"][-1]["retained_accuracy"])
        with self.assertRaises(ValueError):
            validate_probabilities(np.arange(9), p[:, ::-1] * 0.5)

    def test_selection_uses_development_only(self):
        rng = np.random.default_rng(42)
        y = np.repeat(np.arange(9), 20)
        x = np.eye(9)[y] + rng.normal(0, 0.02, (180, 9))
        split = np.array(["train", "validation"] * 90)
        classifier, selected = select_classifier(x, y, split)
        self.assertEqual(len(selected["search"]), 6)
        self.assertEqual(selected["C"], 0.001)  # Exact tie -> smallest C.
        self.assertEqual(classifier.predict_proba(x).shape, (180, 9))
        split[0] = "test"
        with self.assertRaises(ValueError):
            select_classifier(x, y, split)

    def test_roundtrip_publish_and_browser_metric_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            run = temp / "run"
            meta, _result = fixture(run)
            validate_run(run)
            bundle([run], temp / "results.zip")
            docs = temp / "docs"
            (docs / "data").mkdir(parents=True)
            write_json(
                docs / "data/models.json",
                {"schema_version": 1, "models": [], "default": None},
            )
            publish([run], docs)
            publish([run], docs)  # Idempotent import.
            registry = json.loads((docs / "data/models.json").read_text())
            self.assertEqual(registry["default"], meta["run_id"])
            self.assertEqual(len(registry["models"]), 1)
            # Independent JS implementation must reproduce all Python metrics, not only the 54 legacy values.
            script = """import fs from 'node:fs';
import {parseCSV,preparePredictions,checkBenchmarks} from './docs/metrics.mjs';
import {validateExtended} from './docs/benchmark.mjs';
const root=process.argv[1];
const read=f=>fs.readFileSync(root+'/'+f,'utf8');
const meta=JSON.parse(read('run_metadata.json'));
const rows=preparePredictions(parseCSV(read('predictions.csv')),meta);
validateExtended(rows,JSON.parse(read('metrics.json')));
if(checkBenchmarks(rows,parseCSV(read('model_metrics.csv')),meta).some(c=>!c.matches)) throw Error('fixed checks');
"""
            subprocess.run(
                ["node", "--input-type=module", "-e", script, str(run)],
                cwd=ROOT,
                check=True,
            )
            changed = json.loads((run / "metrics.json").read_text())
            changed["accuracy"] = 0.1
            write_json(run / "metrics.json", changed)
            with self.assertRaisesRegex(ValueError, "Checksum"):
                validate_run(run)
            meta["metrics_sha256"] = file_hash(run / "metrics.json")
            write_json(run / "run_metadata.json", meta)
            with self.assertRaisesRegex(ValueError, "Metric value"):
                validate_run(run)

    def test_historical_probability_metrics_match_independent_browser_code(self):
        frame = pd.read_csv(
            ROOT / "docs/data/predictions.csv", float_precision="round_trip"
        )
        y = frame.true_class.map(dict(zip(CLASSES, range(9)))).to_numpy()
        p = frame[[f"p_{c}" for c in CLASSES]].to_numpy()
        with tempfile.TemporaryDirectory() as temp:
            saved = Path(temp) / "metrics.json"
            write_json(saved, metrics(y, p))
            script = """import fs from 'node:fs';
import {parseCSV,preparePredictions} from './docs/metrics.mjs';
import {validateExtended} from './docs/benchmark.mjs';
const rows=preparePredictions(parseCSV(fs.readFileSync('docs/data/predictions.csv','utf8')),
JSON.parse(fs.readFileSync('docs/data/run_metadata.json','utf8')));
validateExtended(rows,JSON.parse(fs.readFileSync(process.argv[1],'utf8')));
"""
            subprocess.run(
                ["node", "--input-type=module", "-e", script, str(saved)],
                cwd=ROOT,
                check=True,
            )

    def test_notebook_is_clean_and_valid(self):
        import nbformat

        notebook = nbformat.read(
            ROOT / "notebooks/01_model_evaluation.ipynb", as_version=4
        )
        nbformat.validate(notebook)
        self.assertEqual(len(notebook.cells), 12)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                # Compile ordinary Python after removing the one IPython pip magic.
                compile(
                    "\n".join(
                        line
                        for line in cell.source.splitlines()
                        if not line.startswith("%")
                    ),
                    "<notebook>",
                    "exec",
                )


if __name__ == "__main__":
    unittest.main()
