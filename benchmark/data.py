"""Reuse the original sample, verify source archives, and freeze shared splits."""

import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from . import CLASSES, VERSION
from .common import digest, file_hash, write_json

REFERENCE = Path(__file__).parent / "manifests"
SOURCE = json.loads((REFERENCE / "source.json").read_text())


def read_reference():
    frames = []
    for name, dataset, split in [
        ("train", "NCT-CRC-HE-100K", "development"),
        ("evaluation", "CRC-VAL-HE-7K", "test"),
    ]:
        f = pd.read_csv(REFERENCE / f"{name}_manifest.csv")
        f["relative_path"] = f.member
        f["class_index"] = f.class_code.map(dict(zip(CLASSES, range(9))))
        f["dataset"], f["split"] = dataset, split
        f["patient_id"], f["slide_id"] = "", ""  # No trustworthy mapping supplied.
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def assign_splits(frame, seed=42, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")
    result = frame.copy()
    dev = result.index[result.split == "development"]
    train, validation = train_test_split(
        dev,
        test_size=validation_fraction,
        stratify=result.loc[dev, "class_code"],
        random_state=seed,
    )
    result.loc[train, "split"] = "train"
    result.loc[validation, "split"] = "validation"
    return result.sort_values("image_id").reset_index(drop=True)


def validate_manifest(frame):
    if not frame.image_id.is_unique or not frame.relative_path.is_unique:
        raise ValueError("Repeated image IDs or paths.")
    if set(frame.split) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test splits.")
    for row in frame.itertuples(index=False):
        p = PurePosixPath(row.relative_path)
        if p.is_absolute() or ".." in p.parts or "\\" in str(p):
            raise ValueError("Unsafe image path.")
        if row.class_code not in CLASSES or row.class_index != CLASSES.index(
            row.class_code
        ):
            raise ValueError("Class mapping mismatch.")
        if (row.split == "test") != (row.dataset == "CRC-VAL-HE-7K"):
            raise ValueError("External test dataset entered development.")
    for split, group in frame.groupby("split"):
        if set(group.class_code) != set(CLASSES):
            raise ValueError(f"Missing tissue classes in {split}.")
    if "sha256" in frame:
        if (
            frame.sha256.isna().any()
            or not frame.sha256.str.fullmatch(r"[0-9a-f]{64}").all()
        ):
            raise ValueError("Missing or invalid image hashes.")
        # Abort rather than silently dropping a duplicate and changing the sample.
        if (frame.groupby("sha256").split.nunique() > 1).any():
            raise ValueError("Exact duplicate image file across splits.")


def download(source, directory):
    import requests

    target = directory / source["archive"]
    if not target.exists():
        partial = target.with_suffix(".zip.part")
        with requests.get(source["url"], stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with (
                partial.open("wb") as out,
                tqdm(
                    total=source["bytes"], unit="B", unit_scale=True, desc=target.name
                ) as bar,
            ):
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        out.write(chunk)
                        bar.update(len(chunk))
        if (
            partial.stat().st_size != source["bytes"]
            or file_hash(partial, "md5") != source["md5"]
        ):
            raise ValueError(f"Archive verification failed: {target.name}")
        partial.replace(target)
    if (
        target.stat().st_size != source["bytes"]
        or file_hash(target, "md5") != source["md5"]
    ):
        raise ValueError(f"Existing archive failed verification: {target}")
    return target


def prepare(work, seed=42, validation_fraction=0.2):
    work = Path(work)
    images, archives = work / "images", work / "archives"
    archives.mkdir(parents=True, exist_ok=True)
    manifest = assign_splits(read_reference(), seed, validation_fraction)
    validate_manifest(manifest)
    for source in SOURCE["sources"].values():
        archive_path = download(source, archives)
        selected = manifest[manifest.archive == source["archive"]]
        with zipfile.ZipFile(archive_path) as archive:
            # Verify the chosen variant against its actual class layout and counts.
            infos = [i for i in archive.infolist() if not i.is_dir()]
            counts = dict.fromkeys(CLASSES, 0)
            for info in infos:
                parts = PurePosixPath(info.filename).parts
                if (
                    len(parts) != 3
                    or parts[1] not in CLASSES
                    or not parts[2].lower().endswith(".tif")
                ):
                    raise ValueError(f"Unexpected archive member: {info.filename}")
                counts[parts[1]] += 1
            if counts != source["counts"]:
                raise ValueError(
                    "Archive tissue counts differ from the documented variant."
                )
            for row in tqdm(
                selected.itertuples(index=False),
                total=len(selected),
                desc="Extract selected patches",
            ):
                info = archive.getinfo(row.member)
                if info.file_size != row.bytes or info.CRC != row.crc32:
                    raise ValueError("Source manifest does not match archive.")
                target = images / row.relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                # zipfile checks CRC while reading, including on reruns.
                with (
                    archive.open(row.member) as src,
                    target.with_suffix(".part").open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
                target.with_suffix(".part").replace(target)
    manifest["sha256"] = [
        file_hash(images / p) for p in tqdm(manifest.relative_path, desc="Audit files")
    ]
    validate_manifest(manifest)
    manifest_path = work / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    spec = {
        "version": VERSION,
        "sampling": "Recovered historical 1000 patches/class; unchanged IDs",
        "sampling_seed": 42,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "split_method": "Stratified patch-level; no documented patient/slide mapping",
        "class_order": CLASSES,
        "source": SOURCE,
        "manifest_sha256": file_hash(manifest_path),
        "source_manifest_hashes": {
            p.name: file_hash(p) for p in REFERENCE.glob("*.csv")
        },
        "split_hashes": {
            s: digest(g.to_dict("records")) for s, g in manifest.groupby("split")
        },
        "duplicate_audit": "Unique IDs and no exact file duplicates across splits",
        "limitations": "File checks cannot establish patient independence or absence of pretraining contamination.",
    }
    write_json(work / "dataset.json", spec)
    return manifest


def load_prepared(work):
    work = Path(work)
    spec = json.loads((work / "dataset.json").read_text())
    if file_hash(work / "manifest.csv") != spec["manifest_sha256"]:
        raise ValueError("Prepared manifest changed; rerun data preparation.")
    frame = pd.read_csv(work / "manifest.csv", keep_default_na=False)
    validate_manifest(frame)
    # Prevent using cached embeddings after a local image has been replaced.
    for row in tqdm(
        frame.itertuples(index=False), total=len(frame), desc="Verify input files"
    ):
        if file_hash(work / "images" / row.relative_path) != row.sha256:
            raise ValueError(f"Image content changed: {row.image_id}")
    return frame, spec
