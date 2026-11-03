"""Batched frozen extraction with verified, content-addressed caches."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .common import digest, file_hash, versions


class Patches:
    def __init__(self, frame, root, transform):
        self.paths = frame.relative_path.tolist()
        self.root, self.transform = Path(root), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.root / self.paths[index]) as image:
            if image.size != (224, 224):
                raise ValueError(f"Unexpected patch dimensions: {self.paths[index]}")
            return self.transform(image.convert("RGB"))


def extract(
    encoder, frame, image_root, cache_dir, batch_size=16, workers=2, precision="fp16"
):
    import torch
    from torch.utils.data import DataLoader

    if batch_size < 1 or workers < 0 or precision not in ("fp16", "fp32"):
        raise ValueError("Invalid extraction settings.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actual_precision = precision if device == "cuda" else "fp32"
    hardware = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    spec = {
        "encoder": encoder.spec,
        "manifest_sha256": digest(frame.to_dict("records")),
        "versions": versions(),
        "precision": actual_precision,
        "device": hardware,
        "batch_size": batch_size,
        "workers": workers,
        "implementation_sha256": digest(
            {
                name: file_hash(Path(__file__).parent / name)
                for name in ("encoders.py", "features.py")
            }
        ),
    }
    path = Path(cache_dir) / f"{digest(spec)}.npz"
    ids = frame.image_id.to_numpy(dtype=str)
    labels = frame.class_index.to_numpy(dtype=np.int64)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if (
                json.loads(z["spec"].item()) != spec
                or not np.array_equal(z["ids"], ids)
                or not np.array_equal(z["labels"], labels)
            ):
                raise ValueError("Embedding cache alignment mismatch.")
            features, measurement = (
                z["features"].copy(),
                json.loads(z["measurement"].item()),
            )
        cache_hit = True
    else:
        try:
            encoder.model.to(device).eval()
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"{encoder.kind}: checkpoint does not fit GPU memory. Use a larger GPU; no substitute weights were loaded."
            ) from exc
        loader = DataLoader(
            Patches(frame, image_root, encoder.transform),
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            pin_memory=device == "cuda",
        )
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        chunks = []
        try:
            with torch.inference_mode():
                for images in tqdm(loader, desc=f"Embed {encoder.kind}"):
                    context = (
                        torch.autocast("cuda", dtype=torch.float16)
                        if actual_precision == "fp16"
                        else nullcontext()
                    )
                    with context:
                        output = encoder.forward(images.to(device, non_blocking=True))
                    if output.ndim != 2 or output.shape[1] != encoder.spec["dimension"]:
                        raise ValueError(
                            "Encoder output shape differs from the documented pooling recipe."
                        )
                    chunks.append(output.float().cpu().numpy())
            if device == "cuda":
                torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError as exc:
            # No silently changed batch sizes, precision or partial caches.
            chunks.clear()
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"{encoder.kind}: GPU memory exhausted. Reduce its batch size (try 1), then rerun; completed caches remain valid."
            ) from exc
        features = np.concatenate(chunks).astype(np.float32)
        measurement = {
            "seconds": time.perf_counter() - started,
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3
            if device == "cuda"
            else None,
            "timing": "Includes data loading, preprocessing, GPU transfer, forward pass and CPU copies; excludes model loading and cache serialization",
            "device": hardware,
            "precision": actual_precision,
        }
        if not np.isfinite(features).all():
            raise ValueError("Non-finite embeddings; rerun this encoder with fp32.")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        with partial.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=features,
                ids=ids,
                labels=labels,
                spec=json.dumps(spec, sort_keys=True),
                measurement=json.dumps(measurement),
            )
        partial.replace(path)
        cache_hit = False
    if (
        features.shape != (len(frame), encoder.spec["dimension"])
        or not np.isfinite(features).all()
    ):
        raise ValueError("Invalid cached embeddings.")
    return features, {
        **measurement,
        "cache_hit": cache_hit,
        "cache_key": path.stem,
        "spec": spec,
    }
