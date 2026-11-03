"""Select C on development validation, refit all development images, then test once."""

import gc
import shutil
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import CLASSES, VERSION
from .common import digest, file_hash, versions, write_json
from .data import load_prepared
from .encoders import ModelUnavailable, load_encoder
from .evaluation import fixed_checks, metrics
from .features import extract

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def fit_classifier(features, labels, c, seed):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs", max_iter=5000, random_state=seed),
    )
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(features, labels)
    return classifier


def select_classifier(features, labels, split, seed=42):
    train, validation = split == "train", split == "validation"
    if (
        not train.any()
        or not validation.any()
        or not np.isin(split, ["train", "validation"]).all()
    ):
        raise ValueError("Classifier selection accepts development partitions only.")
    trials = []
    for c in C_GRID:
        classifier = fit_classifier(features[train], labels[train], c, seed)
        score = f1_score(
            labels[validation],
            classifier.predict(features[validation]),
            labels=range(9),
            average="macro",
            zero_division=0,
        )
        trials.append({"C": c, "validation_macro_f1": float(score), "converged": True})
    # Smaller C wins exact ties; no test data enters selection.
    best = max(trials, key=lambda row: row["validation_macro_f1"])
    classifier = fit_classifier(features, labels, best["C"], seed)
    if not np.array_equal(classifier.classes_, np.arange(9)):
        raise ValueError(
            "Fitted classifier class order differs from canonical probability order."
        )
    return classifier, {
        "C": best["C"],
        "search": trials,
        "criterion": "validation macro F1; smaller C on exact ties",
        "solver": "lbfgs",
        "max_iter": 5000,
        "converged": True,
        "scaler": "fit on train partition during selection, then refit on full development sample",
    }


def run(
    work,
    models=("resnet18", "phikon-v2", "uni2-h"),
    batch_sizes=None,
    workers=2,
    precision="fp16",
    revisions=None,
):
    import torch

    work = Path(work)
    manifest, dataset = load_prepared(work)
    batch_sizes = batch_sizes or {"resnet18": 64, "phikon-v2": 16, "uni2-h": 4}
    dev, test = (
        manifest[manifest.split != "test"].reset_index(drop=True),
        manifest[manifest.split == "test"].reset_index(drop=True),
    )
    if len(dev) != 9000 or len(test) != 7180:
        raise ValueError(
            "This benchmark version requires the recovered 9000/7180 manifests."
        )
    torch.manual_seed(dataset["split_seed"])
    np.random.seed(dataset["split_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    outputs, statuses = [], []
    for name in models:
        encoder = None
        try:
            print(f"\n{name}")
            encoder = load_encoder(name, (revisions or {}).get(name))
            settings = {
                "batch_size": batch_sizes[name],
                "workers": workers,
                "precision": precision,
            }
            x_dev, dev_info = extract(
                encoder, dev, work / "images", work / "cache", **settings
            )
            classifier, selection = select_classifier(
                x_dev,
                dev.class_index.to_numpy(),
                dev.split.to_numpy(),
                dataset["split_seed"],
            )
            protocol = {
                "version": VERSION,
                "model_name": name,
                "review_cutoff": 0.8,
                "tumour_threshold": 0.5,
                "high_confidence": 0.9,
                "dataset": dataset,
                "encoder": encoder.spec,
                "classifier": selection,
                "settings": settings,
                "versions": versions(),
                "implementation_sha256": digest(
                    {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
                ),
            }
            run_id = f"{name}-{digest(protocol)[:16]}"
            target = work / "runs" / run_id
            if target.exists():
                from .publish import validate_run

                validate_run(target)
                outputs.append(target)
                statuses.append({"model": name, "status": "reused", "run_id": run_id})
                print(f"Reusing completed, validated run: {run_id}")
                continue
            # The independent test set is only predicted after selection and final refit.
            x_test, test_info = extract(
                encoder, test, work / "images", work / "cache", **settings
            )
            p = classifier.predict_proba(x_test)
            result = metrics(test.class_index.to_numpy(), p)
            destination = target
            target = destination.parent / ".staging" / run_id
            if target.exists():
                shutil.rmtree(target)  # Only an interrupted, unpublished staging run.
            target.mkdir(parents=True)
            predicted = p.argmax(1)
            frame = pd.DataFrame(
                {
                    "image_id": test.image_id,
                    "dataset_split": "held_out_evaluation",
                    "model_name": name,
                    "run_id": run_id,
                    "true_class": test.class_code,
                    "predicted_class": [CLASSES[i] for i in predicted],
                    "confidence": p.max(1),
                    "is_correct": (predicted == test.class_index.to_numpy()).astype(
                        int
                    ),
                    "is_tumour_true": (test.class_index.to_numpy() == 8).astype(int),
                }
            )
            for i, code in enumerate(CLASSES):
                frame[f"p_{code}"] = p[:, i]
            frame.to_csv(target / "predictions.csv", index=False, float_format="%.17g")
            fixed_checks(test.class_index.to_numpy(), p, result).to_csv(
                target / "model_metrics.csv", index=False, float_format="%.17g"
            )
            write_json(target / "metrics.json", result)
            extraction = {
                "seconds": dev_info["seconds"] + test_info["seconds"],
                "peak_gpu_memory_gb": max(
                    v
                    for v in [
                        dev_info["peak_gpu_memory_gb"],
                        test_info["peak_gpu_memory_gb"],
                    ]
                    if v is not None
                )
                if torch.cuda.is_available()
                else None,
                "development": dev_info,
                "test": test_info,
            }
            try:
                source_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                source_commit = None
            metadata = {
                "schema_version": 1,
                "benchmark_version": VERSION,
                "run_id": run_id,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "train_images": len(dev),
                "evaluated_images": len(test),
                "classifier_probability_order": CLASSES,
                "evaluation_counts": {
                    c: int((test.class_code == c).sum()) for c in CLASSES
                },
                "predictions_sha256": file_hash(target / "predictions.csv"),
                "metrics_sha256": file_hash(target / "metrics.json"),
                "protocol": protocol,
                "extraction": extraction,
                "source_commit": source_commit,
                "hardware": {"device": test_info["device"], "cuda": torch.version.cuda},
                "probabilities": "uncalibrated",
            }
            write_json(target / "run_metadata.json", metadata)
            manifest.to_csv(target / "manifest.csv", index=False)
            joblib.dump(
                classifier, target / "classifier.joblib"
            )  # Local artifact, not in public bundle.
            from .publish import validate_run

            validate_run(target)
            target.rename(destination)
            target = destination
            outputs.append(target)
            statuses.append({"model": name, "status": "complete", "run_id": run_id})
            print(f"Macro F1 {result['macro_f1']:.4f} · {target}")
        except ModelUnavailable as exc:
            statuses.append(
                {"model": name, "status": "unavailable", "reason": str(exc)}
            )
            print(str(exc))
        finally:
            if encoder is not None:
                del encoder
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_json(work / "status.json", statuses)
    return outputs
