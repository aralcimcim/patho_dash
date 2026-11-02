# %% [markdown]
# # German Pathology AI — Model Evaluation Dashboard
# Upload `01_model_evaluation.ipynb` to https://colab.research.google.com/.
# Run cells in order. Optional: Runtime → Change runtime type → T4 GPU.
# This notebook downloads about 12.5 GB of archives; extraction is limited to
# sampled training patches and the complete held-out set. Allow ample disk space.
# Outputs are patch-level research results, not patient diagnoses.
# No performance results have been precomputed or supplied in this notebook.

# %% [markdown]
# ## 1. Dependencies
# Keep Colab's paired torch/torchvision GPU installation. Pin the small classifier
# dependency and record all resolved versions. If Colab requests a restart, restart
# and rerun from this cell. No local Python installation is needed.

# %%
get_ipython().run_line_magic("pip", "install -q scikit-learn==1.7.2 'requests>=2.31,<3' 'tqdm>=4.66,<5'")

# %% [markdown]
# ## 2. Imports, device, and reproducibility
# Seeds control sampling and the classifier. Floating-point results can still
# differ slightly across GPU models and package versions. No augmentation or AMP.

# %%
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torchvision
from PIL import Image
from IPython.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == "cuda":
    print(torch.cuda.get_device_name(0))
else:
    print("CPU fallback: the same baseline runs, but extraction can take longer.")

# %% [markdown]
# ## 3. Configuration and optional Drive persistence
# Set these BEFORE any held-out predictions. Default: 1,000 training images/class,
# batch 32, fixed C=1, review cutoff 0.80, tumour threshold 0.50. No tuning.
# If CUDA runs out of memory, restart and reduce BATCH_SIZE to 16 or 8.
# USE_DRIVE=True mounts your Drive; keep images/archives in temporary /content.
# Drive preserves caches and run artifacts between Colab sessions. With False,
# download the ZIP before ending the runtime; Colab local storage is temporary.

# %%
TRAIN_PER_CLASS = 1000
BATCH_SIZE = 32
NUM_WORKERS = 2
USE_DRIVE = False
MODEL_NAME = "resnet18_imagenet1k_v1_logreg_c1"
REVIEW_CUTOFF = 0.80
TUMOUR_THRESHOLD = 0.50
HIGH_CONFIDENCE = 0.90
CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
assert isinstance(TRAIN_PER_CLASS, int) and TRAIN_PER_CLASS > 0
assert BATCH_SIZE > 0 and NUM_WORKERS >= 0

ROOT = Path("/content/german_pathology_ai")
PERSIST = ROOT
if USE_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    PERSIST = Path("/content/drive/MyDrive/german_pathology_ai")
ARCHIVE_DIR = ROOT / "archives"
IMAGE_DIR = ROOT / "images"
CACHE_DIR = PERSIST / "embedding_cache"
for directory in [ARCHIVE_DIR, IMAGE_DIR, CACHE_DIR, PERSIST / "runs"]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "train": {
        "archive": "NCT-CRC-HE-100K.zip",
        "url": "https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K.zip/content",
        "bytes": 11690284003,
        "md5": "6fd702d11df6292bc054397ae038a464",
        "counts": dict(zip(CLASSES, [10407, 10566, 11512, 11557, 8896, 13536, 8763, 10446, 14317])),
    },
    "evaluation": {
        "archive": "CRC-VAL-HE-7K.zip",
        "url": "https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content",
        "bytes": 800276929,
        "md5": "2fd1651b4f94ebd818ebf90ad2b6ce06",
        "counts": dict(zip(CLASSES, [1338, 847, 339, 634, 1035, 592, 741, 421, 1233])),
    },
}
CLASS_NAMES = [
    ("ADI", "Adipose tissue", "Fettgewebe"),
    ("BACK", "Background", "Hintergrund"),
    ("DEB", "Debris", "Zell- und Gewebetrümmer"),
    ("LYM", "Lymphocytes", "Lymphozyten"),
    ("MUC", "Mucus", "Schleim"),
    ("MUS", "Smooth muscle", "Glatte Muskulatur"),
    ("NORM", "Normal colon mucosa", "Normale Kolonschleimhaut"),
    ("STR", "Cancer-associated stroma", "Tumorassoziiertes Stroma"),
    ("TUM", "Colorectal adenocarcinoma epithelium", "Kolorektales Adenokarzinomepithel"),
]

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

def file_hash(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()

VERSIONS = {name: importlib.metadata.version(name) for name in [
    "torch", "torchvision", "numpy", "pandas", "scikit-learn", "Pillow",
    "requests", "tqdm", "matplotlib", "joblib", "threadpoolctl",
]}
VERSIONS["python"] = platform.python_version()
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1
TRANSFORM = WEIGHTS.transforms()
PROTOCOL = {
    "seed": SEED, "train_per_class": TRAIN_PER_CLASS, "class_codes": CLASSES,
    "model_name": MODEL_NAME, "encoder_weights": str(WEIGHTS),
    "weights_url": WEIGHTS.url, "preprocessing": repr(TRANSFORM),
    "embedding_dim": 512, "inference_dtype": "float32", "batch_size": BATCH_SIZE,
    "device": str(DEVICE),
    "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "num_workers": NUM_WORKERS, "classifier": {"C": 1.0, "solver": "lbfgs", "max_iter": 2000},
    "scaler": "StandardScaler fitted on training embeddings only",
    "review_cutoff": REVIEW_CUTOFF, "tumour_threshold": TUMOUR_THRESHOLD,
    "high_confidence": HIGH_CONFIDENCE, "sources": SOURCES, "versions": VERSIONS,
    "evaluation_policy": "Fixed baseline; complete CRC-VAL-HE-7K held out; no model or threshold selection",
}
PROTOCOL_HASH = digest(PROTOCOL)
RUN_DIR = PERSIST / "runs" / PROTOCOL_HASH[:16]
RUN_DIR.mkdir(exist_ok=True)
protocol_path = RUN_DIR / "frozen_protocol.json"
if protocol_path.exists():
    assert json.loads(protocol_path.read_text())["protocol"] == PROTOCOL
else:
    write_json(protocol_path, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                               "protocol_sha256": PROTOCOL_HASH, "protocol": PROTOCOL})
print("Frozen protocol:", PROTOCOL_HASH, "\nRun folder:", RUN_DIR)
print(TRANSFORM)

# %% [markdown]
# ## 4. Download only the two required archives
# Verified against the original Zenodo record (DOI 10.5281/zenodo.1214456), CC BY 4.0.
# Completed downloads are verified by size and MD5 and reused. A partial download
# is restarted; incomplete files never replace verified archives. Never download NONORM.
# Archive transfer is still ~12.5 GB even though extraction below is selective.

# %%
def download_archive(source, directory):
    target = Path(directory) / source["archive"]
    if target.exists():
        if target.stat().st_size == source["bytes"] and file_hash(target, "md5") == source["md5"]:
            print("Verified; reusing", target.name)
            return target
        raise ValueError(f"Existing archive failed integrity check: {target}. Remove it and rerun this cell.")
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(source["url"], stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with open(partial, "wb") as output, tqdm(total=source["bytes"], unit="B", unit_scale=True,
                                                desc=target.name) as progress:
            for chunk in response.iter_content(8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    progress.update(len(chunk))
    assert partial.stat().st_size == source["bytes"], "Incomplete archive"
    assert file_hash(partial, "md5") == source["md5"], "Archive checksum mismatch"
    partial.replace(target)
    return target

archive_paths = {split: download_archive(source, ARCHIVE_DIR) for split, source in SOURCES.items()}

# %% [markdown]
# ## 5. Audit archive contents and grouping information
# The original record describes 86 training slides and 50 different evaluation
# patients, with no patient overlap. Slides are not patient counts. The two ZIP
# directories contain images and folders, with no documented patient/slide mapping.
# Some filenames contain “TCGA”; this is not sufficient to infer grouping IDs.
# We therefore use no additional validation split or hyperparameter search.
# Random patch splits could put correlated same-slide images on both sides.

# %%
def inspect_archive(path, expected_counts):
    records, other_files = [], []
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():
                continue
            parts = PurePosixPath(item.filename).parts
            if len(parts) != 3 or parts[1] not in CLASSES or not parts[2].lower().endswith(".tif"):
                other_files.append(item.filename)
                continue
            assert not PurePosixPath(item.filename).is_absolute() and ".." not in parts
            records.append({"image_id": f"{Path(path).name}/{item.filename}",
                            "member": item.filename, "class_code": parts[1],
                            "archive": Path(path).name, "bytes": item.file_size, "crc32": item.CRC})
    frame = pd.DataFrame(records).sort_values("image_id").reset_index(drop=True)
    assert frame.image_id.is_unique
    counts = frame.class_code.value_counts().reindex(CLASSES, fill_value=0).to_dict()
    assert counts == expected_counts, (counts, expected_counts)
    assert not other_files, f"Unexpected files: inspect for possible grouping metadata first: {other_files[:10]}"
    return frame, {"counts": counts, "non_image_files": other_files,
                   "grouping_available": False, "grouping_note": "No documented patient/slide mapping in archive or record"}

train_all, train_audit = inspect_archive(archive_paths["train"], SOURCES["train"]["counts"])
eval_manifest, eval_audit = inspect_archive(archive_paths["evaluation"], SOURCES["evaluation"]["counts"])
display(pd.DataFrame({"train_available": train_audit["counts"], "held_out": eval_audit["counts"]}))
write_json(RUN_DIR / "grouping_audit.json", {"train": train_audit, "evaluation": eval_audit})
print("No grouping mapping available: fixed baseline, no random patch validation split.")

# %% [markdown]
# ## 6. Reproducible training subsampling and selective extraction
# Sample from sorted training filenames using seed 42. Keep every evaluation patch.
# Stable image IDs include archive name and original relative image path.
# Keep the ZIPs to make restarts cheaper; do not extract all 100,000 training images.

# %%
def sample_training(frame, limit, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for code in CLASSES:
        group = frame.loc[frame.class_code == code].sort_values("image_id")
        positions = rng.choice(len(group), size=min(limit, len(group)), replace=False)
        selected.append(group.iloc[positions])
    return pd.concat(selected).sort_values("image_id").reset_index(drop=True)

def extract_selected(path, manifest, destination):
    with zipfile.ZipFile(path) as archive:
        for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc=f"Extract {Path(path).name}"):
            parts = PurePosixPath(row.member).parts
            assert not PurePosixPath(row.member).is_absolute() and ".." not in parts
            target = Path(destination).joinpath(*parts)
            if target.exists() and target.stat().st_size == row.bytes:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(row.member) as source, open(temporary, "wb") as output:
                shutil.copyfileobj(source, output)
            assert temporary.stat().st_size == row.bytes
            temporary.replace(target)

train_manifest = sample_training(train_all, TRAIN_PER_CLASS, SEED)
assert set(train_manifest.image_id).isdisjoint(eval_manifest.image_id)
for split, manifest in [("train", train_manifest), ("evaluation", eval_manifest)]:
    manifest.to_csv(RUN_DIR / f"{split}_manifest.csv", index=False)
    extract_selected(archive_paths[split], manifest, IMAGE_DIR)
display(train_manifest.class_code.value_counts().reindex(CLASSES).rename("sampled_train_count"))
print("Training:", len(train_manifest), "Held-out evaluation:", len(eval_manifest))

# %% [markdown]
# ## 7. Inspect a small training image grid; load images
# The grid uses training images only. The frozen ImageNet transform converts RGB
# pixels to tensors, resizes the shorter edge to 256, center-crops 224×224, and
# applies ImageNet normalization. This adds no fitted preprocessing on evaluation.

# %%
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax, code in zip(axes.flat, CLASSES):
    row = train_manifest.loc[train_manifest.class_code == code].iloc[0]
    with Image.open(IMAGE_DIR / row.member) as image:
        assert image.size == (224, 224), (row.image_id, image.size)
        ax.imshow(image.convert("RGB"))
    ax.set_title(code)
    ax.axis("off")
fig.tight_layout()
plt.show()
plt.close(fig)

class PatchDataset(Dataset):
    def __init__(self, manifest, root, transform):
        self.members = manifest.member.tolist()
        self.root, self.transform = Path(root), transform

    def __len__(self):
        return len(self.members)

    def __getitem__(self, index):
        with Image.open(self.root / self.members[index]) as image:
            if image.size != (224, 224):
                raise ValueError(f"Unexpected image size: {self.members[index]} {image.size}")
            return self.transform(image.convert("RGB"))

# %% [markdown]
# ## 8. Frozen ResNet-18 encoder
# The final ImageNet classification layer becomes Identity, yielding 512 features.
# This trains only a small linear classifier, making a manageable first baseline.
# The ImageNet encoder was not trained specifically for pathology; no accuracy is promised.

# %%
encoder = resnet18(weights=WEIGHTS)
encoder.fc = torch.nn.Identity()
encoder.requires_grad_(False)
encoder.eval().to(DEVICE)
weights_file = Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS.url).name
WEIGHTS_SHA256 = file_hash(weights_file)
assert WEIGHTS_SHA256.startswith("f37072fd"), "Unexpected pretrained checkpoint"
print("Encoder checkpoint SHA256:", WEIGHTS_SHA256)

# %% [markdown]
# ## 9. Extract and cache embeddings
# Cache keys cover source checksum, exact manifest, weights hash, transform, versions,
# batch size and device. A restart with the same settings reuses matching .npz files.
# With USE_DRIVE=False, caches survive only while the Colab runtime storage survives.

# %%
def embedding_spec(manifest, split):
    return {
        "manifest_sha256": digest(manifest.to_dict(orient="records")), "split": split,
        "archive_md5": SOURCES[split]["md5"], "weights_sha256": WEIGHTS_SHA256,
        "transform": repr(TRANSFORM), "torch": VERSIONS["torch"],
        "torchvision": VERSIONS["torchvision"], "pillow": VERSIONS["Pillow"],
        "dtype": "float32", "batch_size": BATCH_SIZE, "device": str(DEVICE),
        "device_name": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else platform.processor(),
    }

def extract_embeddings(manifest, split):
    spec = embedding_spec(manifest, split)
    cache_path = CACHE_DIR / f"{split}_{digest(spec)}.npz"
    ids = manifest.image_id.to_numpy(dtype=str)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            assert json.loads(cached["spec"].item()) == spec
            assert np.array_equal(cached["image_ids"], ids)
            features = cached["embeddings"].copy()
        print("Reusing", cache_path.name)
    else:
        dataset = PatchDataset(manifest, IMAGE_DIR, TRANSFORM)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
        blocks = []
        encoder.eval()
        with torch.inference_mode():
            for images in tqdm(loader, desc=f"Embed {split}"):
                blocks.append(encoder(images.to(DEVICE, non_blocking=True)).cpu().numpy())
        features = np.concatenate(blocks).astype(np.float32, copy=False)
        temporary = cache_path.with_suffix(".part")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, embeddings=features, image_ids=ids,
                                spec=np.array(json.dumps(spec, sort_keys=True)))
        temporary.replace(cache_path)
    assert features.shape == (len(manifest), 512) and features.dtype == np.float32
    assert np.isfinite(features).all()
    return features, {"cache_file": cache_path.name, "spec": spec}

X_train, train_embedding_info = extract_embeddings(train_manifest, "train")
y_train = train_manifest.class_code.to_numpy()

# %% [markdown]
# ## 10. Fit the training-only scaler and fixed logistic regression
# No evaluation embeddings or labels enter fit. C=1 and max_iter=2000 are fixed.
# If a convergence warning appears, retain it in your run notes; do not tune settings
# after inspecting held-out results. Changing the baseline requires a new protocol.

# %%
classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=SEED)),
])
with threadpool_limits(limits=1):
    classifier.fit(X_train, y_train)
CLASS_ORDER = classifier.named_steps["logreg"].classes_.tolist()
assert set(CLASS_ORDER) == set(CLASSES)
joblib.dump(classifier, RUN_DIR / "classifier.joblib")
write_json(RUN_DIR / "class_mapping.json", {
    "classifier_probability_order": CLASS_ORDER,
    "probability_column_for_class": {code: f"p_{code}" for code in CLASSES},
    "encoder_weights": str(WEIGHTS), "encoder_checkpoint_sha256": WEIGHTS_SHA256,
})
print("Training-only fit complete. Classifier probability order:", CLASS_ORDER)
print("Optimizer iterations:", classifier.named_steps["logreg"].n_iter_)

# %% [markdown]
# ## 11. Predict on the complete, independent held-out set
# First model predictions on CRC-VAL-HE-7K occur here, after freezing the protocol.
# Do not choose a model or threshold based on these results. Subsequent threshold
# exploration in Power BI describes this set only and is explicitly exploratory.

# %%
assert digest(PROTOCOL) == PROTOCOL_HASH, "Protocol changed after freezing"
X_eval, eval_embedding_info = extract_embeddings(eval_manifest, "evaluation")
with threadpool_limits(limits=1):
    probabilities = classifier.predict_proba(X_eval)
y_eval = eval_manifest.class_code.to_numpy()
predicted = np.asarray(CLASS_ORDER)[probabilities.argmax(axis=1)]
predictions = pd.DataFrame({
    "image_id": eval_manifest.image_id.to_numpy(), "dataset_split": "held_out_evaluation",
    "model_name": MODEL_NAME, "true_class": y_eval, "predicted_class": predicted,
    "confidence": probabilities.max(axis=1), "is_correct": (y_eval == predicted).astype(int),
    "is_tumour_true": (y_eval == "TUM").astype(int),
})
for code in CLASSES:
    predictions[f"p_{code}"] = probabilities[:, CLASS_ORDER.index(code)]

# %% [markdown]
# ## 12. Metrics and confusion matrix
# Macro F1 always averages the fixed nine classes (zero division → 0 for F1).
# Undefined precision/recall is blank, not an invented zero. No patient-level CI.

# %%
def validate_predictions(frame, expected_ids, class_order):
    required = ["image_id", "dataset_split", "model_name", "true_class", "predicted_class",
                "confidence", "is_correct", "is_tumour_true"] + [f"p_{c}" for c in CLASSES]
    assert frame.columns.tolist() == required
    assert len(frame) == len(expected_ids) and len(frame) > 0
    assert frame.image_id.is_unique and set(frame.image_id) == set(expected_ids)
    assert not frame[required].isna().any().any()
    assert frame.dataset_split.eq("held_out_evaluation").all()
    assert frame.model_name.nunique() == 1
    assert set(frame.true_class).issubset(CLASSES) and set(frame.predicted_class).issubset(CLASSES)
    assert len(class_order) == 9 and set(class_order) == set(CLASSES)
    p = frame[[f"p_{c}" for c in class_order]].to_numpy(dtype=float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p.sum(axis=1), 1, atol=1e-8, rtol=0)
    assert np.array_equal(frame.predicted_class.to_numpy(), np.asarray(class_order)[p.argmax(axis=1)])
    assert np.allclose(frame.confidence, p.max(axis=1), atol=1e-10, rtol=0)
    assert np.array_equal(frame.is_correct.to_numpy(), (frame.true_class == frame.predicted_class).astype(int))
    assert np.array_equal(frame.is_tumour_true.to_numpy(), (frame.true_class == "TUM").astype(int))

def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None

def metric_table(frame, review_cutoff, tumour_threshold, high_confidence):
    rows = []
    def add(name, value, code="ALL", threshold=None, numerator=None, denominator=None):
        rows.append(dict(metric_name=name, class_code=code, threshold=threshold,
                         value=value, numerator=numerator, denominator=denominator))
    truth, pred = frame.true_class.to_numpy(), frame.predicted_class.to_numpy()
    add("evaluated_images", len(frame))
    add("correct_predictions", int(frame.is_correct.sum()))
    add("accuracy", accuracy_score(truth, pred), numerator=int(frame.is_correct.sum()), denominator=len(frame))
    add("macro_f1", f1_score(truth, pred, labels=CLASSES, average="macro", zero_division=0))
    for code in CLASSES:
        actual, selected = truth == code, pred == code
        tp, a, p = int((actual & selected).sum()), int(actual.sum()), int(selected.sum())
        add("recall", safe_ratio(tp, a), code, numerator=tp, denominator=a)
        add("precision", safe_ratio(tp, p), code, numerator=tp, denominator=p)
        add("f1", safe_ratio(2 * tp, a + p) if a + p else 0.0, code, numerator=2 * tp, denominator=a + p)
        add("support", a, code)
    positive = frame.p_TUM.to_numpy() >= tumour_threshold
    actual = frame.is_tumour_true.to_numpy().astype(bool)
    tp, fp = int((positive & actual).sum()), int((positive & ~actual).sum())
    tn, fn = int((~positive & ~actual).sum()), int((~positive & actual).sum())
    for name, count in [("tumour_tp", tp), ("tumour_fp", fp), ("tumour_tn", tn), ("tumour_fn", fn)]:
        add(name, count, threshold=tumour_threshold)
    for name, num, den in [("tumour_sensitivity", tp, tp + fn), ("tumour_specificity", tn, tn + fp),
                           ("tumour_precision", tp, tp + fp)]:
        add(name, safe_ratio(num, den), threshold=tumour_threshold, numerator=num, denominator=den)
    review = frame.confidence.to_numpy() < review_cutoff
    retained = ~review
    nr, nk = int(review.sum()), int(retained.sum())
    correct_retained = int(frame.loc[retained, "is_correct"].sum())
    add("review_images", nr, threshold=review_cutoff)
    add("review_proportion", safe_ratio(nr, len(frame)), threshold=review_cutoff, numerator=nr, denominator=len(frame))
    add("coverage", safe_ratio(nk, len(frame)), threshold=review_cutoff, numerator=nk, denominator=len(frame))
    add("retained_accuracy", safe_ratio(correct_retained, nk), threshold=review_cutoff,
        numerator=correct_retained, denominator=nk)
    high = frame.confidence.to_numpy() >= high_confidence
    nh, ne = int(high.sum()), int((high & (frame.is_correct.to_numpy() == 0)).sum())
    add("high_confidence_images", nh, threshold=high_confidence)
    add("high_confidence_errors", ne, threshold=high_confidence)
    add("high_confidence_error_rate", safe_ratio(ne, nh), threshold=high_confidence, numerator=ne, denominator=nh)
    return pd.DataFrame(rows)

validate_predictions(predictions, eval_manifest.image_id.tolist(), CLASS_ORDER)
metrics = metric_table(predictions, REVIEW_CUTOFF, TUMOUR_THRESHOLD, HIGH_CONFIDENCE)
display(metrics.loc[metrics.class_code == "ALL", ["metric_name", "threshold", "value"]])
display(metrics.loc[metrics.class_code != "ALL"].pivot(index="class_code", columns="metric_name", values="value"))
matrix = confusion_matrix(y_eval, predicted, labels=CLASSES)
assert matrix.shape == (9, 9) and matrix.sum() == len(eval_manifest)
fig, ax = plt.subplots(figsize=(9, 8))
ConfusionMatrixDisplay(matrix, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Fixed baseline — held-out image patch counts")
fig.tight_layout()
fig.savefig(RUN_DIR / "confusion_matrix.png", dpi=160)
plt.show()
plt.close(fig)

# %% [markdown]
# ## 13. Fixed tumour-versus-other-tissue benchmark (0.50)
# Other tissue classes include background, debris, stroma and other categories;
# this is not a tumour-versus-benign diagnosis. Confidence scores are uncalibrated.

# %%
display(metrics.loc[metrics.metric_name.str.startswith("tumour_")])
assert int(metrics.loc[metrics.metric_name.isin(["tumour_tp", "tumour_fp", "tumour_tn", "tumour_fn"]), "value"].sum()) == len(eval_manifest)

# %% [markdown]
# ## 14. Export, round-trip validation, provenance, and model artifacts
# Empty metric values represent undefined ratios. CSVs are UTF-8; probabilities use
# enough digits for validation after reloading. Role-specific class dimensions and
# independent parameter tables support the browser-only Power BI model.

# %%
def write_lookup_tables(directory):
    directory = Path(directory)
    lookup = pd.DataFrame(CLASS_NAMES, columns=["class_code", "english_name", "german_name"])
    for name in ["class_lookup", "predicted_classes"]:
        lookup.to_csv(directory / f"{name}.csv", index=False)
    thresholds = pd.DataFrame({"threshold": np.arange(101) / 100})
    for name in ["thresholds", "review_thresholds", "tumour_thresholds"]:
        thresholds.to_csv(directory / f"{name}.csv", index=False, float_format="%.2f")
    bins = pd.DataFrame({"bin_order": np.arange(10), "lower": np.arange(10) / 10,
                         "upper": np.arange(1, 11) / 10,
                         "bin_label": [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)]})
    bins.to_csv(directory / "confidence_bins.csv", index=False, float_format="%.1f")

predictions.to_csv(RUN_DIR / "predictions.csv", index=False, float_format="%.12g")
reloaded = pd.read_csv(RUN_DIR / "predictions.csv")
validate_predictions(reloaded, eval_manifest.image_id.tolist(), CLASS_ORDER)
roundtrip_metrics = metric_table(reloaded, REVIEW_CUTOFF, TUMOUR_THRESHOLD, HIGH_CONFIDENCE)
assert np.allclose(metrics.value.to_numpy(float), roundtrip_metrics.value.to_numpy(float), equal_nan=True)
metrics.to_csv(RUN_DIR / "model_metrics.csv", index=False, float_format="%.12g")
pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).rename_axis("true_class").to_csv(RUN_DIR / "confusion_matrix.csv")
write_lookup_tables(RUN_DIR)
metadata = {
    "project": "German Pathology AI — Model Evaluation Dashboard",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_HASH,
    "protocol": PROTOCOL, "classifier_probability_order": CLASS_ORDER,
    "pretrained_checkpoint_sha256": WEIGHTS_SHA256,
    "train_embedding_cache": train_embedding_info, "evaluation_embedding_cache": eval_embedding_info,
    "train_images": len(train_manifest), "evaluated_images": len(eval_manifest),
    "train_counts": {c: int((train_manifest.class_code == c).sum()) for c in CLASSES},
    "evaluation_counts": {c: int((eval_manifest.class_code == c).sum()) for c in CLASSES},
    "training_manifest_sha256": file_hash(RUN_DIR / "train_manifest.csv"),
    "evaluation_manifest_sha256": file_hash(RUN_DIR / "evaluation_manifest.csv"),
    "predictions_sha256": file_hash(RUN_DIR / "predictions.csv"),
    "device": str(DEVICE), "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
    "classifier_iterations": classifier.named_steps["logreg"].n_iter_.tolist(),
    "classifier_parameters": classifier.named_steps["logreg"].get_params(),
    "scaler_parameters": classifier.named_steps["scaler"].get_params(),
    "classifier_converged": bool((classifier.named_steps["logreg"].n_iter_ < 2000).all()),
    "dataset_citation": "Kather, Jakob Nikolas; Halama, Niels; Marx, Alexander (2018). 100,000 histological images of human colorectal cancer and healthy tissue (v0.1). Zenodo. https://doi.org/10.5281/zenodo.1214456",
    "dataset_record": "https://zenodo.org/records/1214456", "dataset_license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "evaluation_notes": [
        "Final held-out archive was never used for fitting, threshold selection, or model selection.",
        "No documented patient/slide mapping was available; no random patch-level validation split was used.",
        "Original record reports 86 training slides and 50 independent evaluation patients; patches within a patient are correlated.",
        "The selected training sample is not 9,000 independent patients; this is patch-level evaluation.",
        "Interactive threshold changes are exploratory, not optimal or clinically validated operating points.",
        "Other tissue classes include background; do not interpret them all as benign.",
        "ImageNet/logistic regression probabilities are not calibrated clinical risks.",
        "Dataset record explicitly describes normalized training images; no extra claim of evaluation normalization is made.",
    ],
}
write_json(RUN_DIR / "run_metadata.json", metadata)
(RUN_DIR / "resolved_versions.txt").write_text("\n".join(f"{k}=={v}" for k, v in VERSIONS.items()) + "\n")
(RUN_DIR / "DATA_ATTRIBUTION.txt").write_text(metadata["dataset_citation"] + "\nDataset: CC BY 4.0.\n"
    "Derived outputs: embeddings classified with frozen ImageNet ResNet18 and training-only logistic regression.\n"
    "No original images are included. Outputs are patch-level research predictions, not patient diagnoses.\n")
print("Export validation passed:", len(reloaded), "unique held-out patches.")

# %% [markdown]
# ## 15. Download the small deliverables ZIP
# Contains CSVs, manifests, provenance, fitted classifier and confusion figure.
# Excludes all original images, archives, embedding caches, and ResNet checkpoints.
# Classifier.joblib must only be loaded from your own trusted runs. Restore the
# encoder using the recorded weights enum; retain exact versions for reproduction.
# Next: follow powerbi/SETUP.md from the project folder, using these actual outputs.

# %%
deliverables = sorted(path for path in RUN_DIR.iterdir() if path.is_file())
assert not any(path.suffix.lower() in {".tif", ".tiff", ".jpg", ".npz", ".pth", ".zip"} for path in deliverables)
zip_path = ROOT / f"german_pathology_ai_{PROTOCOL_HASH[:16]}.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in deliverables:
        archive.write(path, arcname=path.name)
print("ZIP:", zip_path, f"({zip_path.stat().st_size / 1e6:.1f} MB)")
from google.colab import files
files.download(str(zip_path))
