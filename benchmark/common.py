"""Small, shared serialization helpers. No model or dataset dependencies."""

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def versions():
    result = {"python": platform.python_version()}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "timm",
        "transformers",
        "huggingface-hub",
        "Pillow",
        "joblib",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
