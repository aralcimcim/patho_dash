"""Optional CPU inference checks; no downloaded models or benchmark scores."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from benchmark.common import file_hash

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "PyTorch is optional for CPU protocol/export tests")
class FeatureTests(unittest.TestCase):
    def test_inference_cache_revision_and_alignment(self):
        import torch

        from benchmark.encoders import Encoder
        from benchmark.features import extract

        class Probe(torch.nn.Module):
            def forward(self, batch):
                if self.training or torch.is_grad_enabled():
                    raise AssertionError("Extraction must use eval and inference_mode.")
                return batch.mean(dim=(2, 3))

        def transform(image):
            return (
                torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()
                / 255
            )

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for i in range(3):
                path = root / f"{i}.tif"
                Image.new("RGB", (224, 224), color=(i * 50, 0, 0)).save(path)
                records.append(
                    {
                        "image_id": str(i),
                        "relative_path": path.name,
                        "class_index": i,
                        "sha256": file_hash(path),
                    }
                )
            frame = pd.DataFrame(records)
            encoder = Encoder(
                Probe(),
                transform,
                {"revision": "test-only-v1", "dimension": 3},
                "test-only",
            )
            first, info = extract(
                encoder, frame, root, root / "cache", batch_size=2, workers=0
            )
            self.assertFalse(info["cache_hit"])
            self.assertEqual(
                info["precision"], "fp32"
            )  # CPU safely overrides requested FP16.
            second, reused = extract(
                encoder, frame, root, root / "cache", batch_size=2, workers=0
            )
            self.assertTrue(reused["cache_hit"])
            np.testing.assert_array_equal(first, second)
            self.assertEqual(info["seconds"], reused["seconds"])
            encoder.spec["revision"] = "test-only-v2"
            _, changed = extract(
                encoder, frame, root, root / "cache", batch_size=2, workers=0
            )
            self.assertNotEqual(info["cache_key"], changed["cache_key"])
            cached = root / "cache" / f"{changed['cache_key']}.npz"
            with np.load(cached, allow_pickle=False) as z:
                contents = dict(z)
            contents["ids"] = np.array(["wrong", "ids", "here"])
            np.savez_compressed(cached, **contents)
            with self.assertRaisesRegex(ValueError, "alignment"):
                extract(encoder, frame, root, root / "cache", batch_size=2, workers=0)

    def test_complete_run_exports_and_reuses_without_new_test_predictions(self):
        from unittest.mock import patch

        import torch

        from benchmark.data import assign_splits, read_reference
        from benchmark.encoders import Encoder, ModelUnavailable
        from benchmark.publish import validate_run
        from benchmark.run import run

        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            manifest = assign_splits(read_reference())
            manifest["sha256"] = [f"{i:064x}" for i in range(len(manifest))]
            manifest.to_csv(work / "manifest.csv", index=False)
            dataset = {
                "manifest_sha256": file_hash(work / "manifest.csv"),
                "split_seed": 42,
            }
            calls = []

            def features(encoder, frame, *args, **kwargs):
                calls.append(frame.split.unique().tolist())
                return np.eye(9)[frame.class_index.to_numpy()].astype(np.float32), {
                    "seconds": 0.01,
                    "peak_gpu_memory_gb": None,
                    "device": "CPU test fixture",
                }

            def encoder(name, revision):
                if name == "uni2-h":
                    raise ModelUnavailable("Test-only missing approval")
                return Encoder(
                    torch.nn.Identity(),
                    None,
                    {"dimension": 9, "revision": "test-only"},
                    "test-only",
                )

            with (
                patch("benchmark.run.load_prepared", return_value=(manifest, dataset)),
                patch("benchmark.run.load_encoder", side_effect=encoder),
                patch("benchmark.run.extract", side_effect=features),
            ):
                outputs = run(work, models=["resnet18", "uni2-h"])
                self.assertEqual(len(outputs), 1)
                validate_run(outputs[0])
                original_calls = len(calls)
                reused = run(work, models=["resnet18"])
                self.assertEqual(reused, outputs)
                # Development features may be reused; test features are not predicted twice.
                self.assertEqual(len(calls), original_calls + 1)
                self.assertNotEqual(calls[-1], ["test"])


if __name__ == "__main__":
    unittest.main()
