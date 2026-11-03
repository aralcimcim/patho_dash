# Dashboard

[Live dashboard](https://aralcimcim.github.io/patho_dash/) · [Open notebook in Colab](https://colab.research.google.com/github/aralcimcim/patho_dash/blob/main/notebooks/01_model_evaluation.ipynb)

A benchmark of **frozen image encoders + logistic regression** on colorectal tissue patches. The static dashboard supports model selection, a sortable macro-F1 leaderboard, aligned error comparisons, confusion matrices, recall, confidence filters, exploratory thresholds and CSV downloads. Additional benchmark details stay collapsed by default.

Only the historical ResNet-18 C=1 evaluation is currently available: **86.38% accuracy, 82.77% macro F1**, tumour sensitivity **82.40%** and specificity **98.35%** at 0.50. These are saved reference results, not new experiments. ResNet-18 under the new protocol, Phikon-v2 and UNI2-h remain explicitly **not run** until genuine exports are published. Missing timing measurements display a dash.

## Run in Colab

Open [`notebooks/01_model_evaluation.ipynb`](notebooks/01_model_evaluation.ipynb), choose **T4 GPU**, and run its six short sections. Keep Colab's paired CUDA torch/torchvision installation. Transformers 5 and Hub ≥1.32 share the Hub version range required by current Colab Gradio/Diffusers packages. The requirements file pins the model-loading libraries and classifier; the run records all resolved versions. If pip requests a runtime restart, restart before continuing.

The notebook downloads ~12.5 GB of verified archives and selectively extracts 16,180 images. Allow ~25 GB free. Drive persists embeddings and completed runs; images/archives stay in temporary Colab storage. Repeated runs reuse compatible caches and immutable completed exports. No images or checkpoints belong in Git.

Start with `MODELS = ["resnet18", "phikon-v2"]`, then add `"uni2-h"`. UNI2-h needs Hugging Face approval; set the Colab secret `HF_TOKEN` and allow notebook access to that secret. Tokens are read from the environment, never written to exports. Missing access skips only that model and records its unavailable status. No random-weight fallback exists. If inference runs out of GPU memory, lower that model's batch size (try 1); an FP16 non-finite feature error requires an explicit FP32 rerun. T4 uses FP16, not BF16. Availability on a particular Colab GPU is not guaranteed.

Equivalent local commands (Python 3.11+, install a matching torch/torchvision pair for your hardware first):

```bash
python -m pip install -r requirements-colab.txt
python -m benchmark prepare --work /path/to/patho_work
python -m benchmark run --work /path/to/patho_work --models resnet18 phikon-v2
python -m benchmark run --work /path/to/patho_work --models uni2-h --batch-size 1
python -m benchmark bundle /path/to/patho_work/runs/*/ --output dashboard-results.zip
```

`run` extracts embeddings, selects the classifier, refits, evaluates and exports. `benchmark/encoders.py`, `features.py`, `run.py`, `evaluation.py` and `publish.py` separate those stages. Pass `--precision fp32` or `--workers 0` when needed. The Python `run(..., revisions={...})` API accepts immutable Hugging Face checkpoint SHAs for replay; otherwise the current checkpoint is resolved once to a recorded SHA.

## Shared protocol

- **Classes:** ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM, in that exact probability-column order.
- **Development:** recovered historical 9,000-image manifest, 1,000 patches per class; original sampling seed 42. A fixed stratified 80/20 patch split (seed 42) supplies classifier training/validation. No trustworthy patient or slide mapping is provided; identifiers are left empty.
- **External test:** the complete, identical 7,180 CRC-VAL-HE-7K IDs for every encoder. It never selects C, thresholds, calibration or other settings.
- **Images:** verified `NCT-CRC-HE-100K.zip` (Macenko-normalized), not NONORM, and original `CRC-VAL-HE-7K.zip`. The [dataset record](https://zenodo.org/records/1214456) explicitly documents training stain normalization; it does not explicitly specify test stain normalization. Both report 224×224 at 0.5 MPP. No extra stain normalization is applied. Channel normalization and resizing do not change stain protocol or physical resolution.
- **Integrity:** source size/MD5, class counts, original ZIP members/CRC, unique IDs/paths and SHA-256 duplicate-file checks across splits. Changed image content invalidates preparation/caches. A duplicate aborts instead of silently changing the sample.
- **Frozen extraction:** eval, inference mode, seeded deterministic algorithms, ordered batches. Cache identity includes checkpoint revision, preprocessing, pooling, exact content-hashed manifest, versions, precision, hardware and batching. IDs and labels are checked on reuse.
- **Linear probe:** training-only StandardScaler + multiclass lbfgs logistic regression. Every encoder searches C `[0.001, 0.01, 0.1, 1, 10, 100]` with validation macro F1; exact ties choose smaller C. Refit scaler and classifier on all 9,000 development patches, then evaluate test once. Non-convergence aborts rather than reporting a completed run. The historical fixed-C baseline remains a separate protocol.
- **Timing:** development + test extraction, including image loading, preprocessing, transfer and CPU feature copies; excludes model/checkpoint loading and cache serialization. Cached runs retain their original measured extraction time. Peak memory is CUDA's maximum allocated memory, including resident model weights, in GiB. It is not total reserved GPU memory. Metadata records GPU, CUDA, precision, workers and batch sizes.

## Encoder recipes and access

The adapters follow official extraction recipes; preprocessing/pooling/dimensions and immutable checkpoint revisions are recorded per run.

| Encoder                                                                                          | Features                                          | Loading and preprocessing                                                                         | Access / terms                                       |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [ResNet-18](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) | 512; global average pool, final FC removed        | torchvision `IMAGENET1K_V1`; documented resize 256 / crop 224, ImageNet channel normalization     | Public torchvision checkpoint; see upstream terms    |
| [Phikon-v2](https://huggingface.co/owkin/phikon-v2)                                              | 1,024; `last_hidden_state[:, 0, :]`               | Transformers AutoModel + checkpoint AutoImageProcessor, both at the same resolved revision        | Owkin non-commercial license; review official terms  |
| [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)                                               | 1,536; CLS output, eight register tokens excluded | Official custom ViT-H/14 architecture, strict checkpoint load; resize 224, ImageNet normalization | Gated, institutional-email approval; CC BY-NC-ND 4.0 |

Future adapters belong in `benchmark/encoders.py`; they are not executable benchmark options yet. Their official recipes differ: [H0-mini](https://huggingface.co/bioptimus/H0-mini) uses SwiGLUPacked/SiLU, checkpoint-derived transforms and its recommended 768-D CLS token; [Virchow2](https://huggingface.co/paige-ai/Virchow2) concatenates CLS and mean patch tokens **after skipping four register tokens** for 2,560 dimensions; [H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) uses its documented 224px/0.5-MPP input, custom channel normalization and 1,536-D output. These models require access approval and have non-commercial terms; consult each linked card before use. Do not substitute a generic timm loader or redistribute weights.

## Metrics and exports

**Primary:** macro F1 over the fixed nine classes. Also: accuracy, balanced accuracy, class precision/recall/F1/support, OVR and macro AUROC, raw/row-normalized confusion matrices, multiclass log loss and Brier score. Brier is `mean_images(sum_classes((p - one_hot_y)^2))`, with no division by nine. Undefined ratios/AUROCs are JSON `null` or empty CSV cells, displayed as a dash.

Tumour-vs-rest uses **p_TUM**, independently of multiclass argmax: AUROC, average precision, sensitivity/specificity/precision/F1 and a labelled `[[TN, FP], [FN, TP]]` matrix at 0.50. Review curves retain confidence ≥ cutoff and report coverage plus retained accuracy/error. Probabilities are **uncalibrated**. Dashboard threshold changes are exploratory and do not select a clinical operating point.

Each immutable `runs/<model>-<protocol-hash>/` has:

- `predictions.csv`: schema v1; stable image ID, dataset split, model/run IDs, true/predicted classes, confidence, correctness/tumour flags and `p_<CODE>` for all nine classes, written at full precision.
- `metrics.json`: aggregate/class/tumour metrics, confusion matrices and 101-point review curve.
- `model_metrics.csv`: the 54 shared browser consistency checks; not an artificial limit on available metrics.
- `run_metadata.json`: schema/benchmark version, source code commit, checkpoint/preprocessing/pooling/dimension, full manifest and split hashes, seeds, C-search results, software/hardware, precision, cache identity and extraction measurements.
- `manifest.csv`: all train/validation/test IDs, relative paths, class indices, dataset/split and exact file hashes. Patient/slide columns are empty when unknown.

A fitted classifier is saved locally for reproducibility but excluded from the public ZIP along with images, embeddings, checkpoints and credentials.

## Publish real results

Extract the notebook's ZIP into a local directory. From this checkout:

```bash
python -m benchmark publish /path/to/extracted/*/
npm test
npm run serve
```

Preview `http://localhost:4179`. The importer validates the original test IDs/labels, manifests, probabilities, argmax, flags, checksums and recomputed metrics **before** updating `docs/data/models.json`. Importing an identical run is idempotent; differing files cannot overwrite a published run ID. New runs live under `docs/data/runs/`; the original historical files stay intact.

After review:

```bash
git add docs/data/models.json docs/data/runs
git commit -m "Add completed encoder benchmark results"
git push
```

GitHub Pages serves `/docs`. All inference runs outside the browser. The browser independently validates probabilities/checksums and recomputes saved metrics, including AUROCs, log loss, Brier and review curves, before showing results. The model picker switches every chart/error table. The sortable leaderboard defaults to macro F1. Pairwise comparison requires the same benchmark version and exact full manifest, aligns stable IDs, and separates corrected errors, new errors, both-wrong cases and disagreements. All case categories can be downloaded. Historical and new protocols cannot be paired.

## Checks

```bash
python -m pip install nbformat  # test-only dependency
python -m unittest discover -s tests -p 'test_*.py' -v
npm ci
npm test
PATHOLOGY_BROWSER_PATH=/usr/bin/google-chrome npm run test:browser
```

Tests cover shared splits, duplicate detection, probability alignment, independent Python/JavaScript metric parity, classifier selection, clean notebook structure, real historical metrics, model switching, ID-aligned comparisons, downloads and responsive layouts. Optional PyTorch CPU checks also exercise inference mode, cache invalidation and completed-run reuse. Synthetic fixtures exist only in tests. The new pretrained GPU extractions and complete benchmark runs have **not** been executed in this checkout.

## Limits and attribution

These are correlated image patches, not patient diagnoses. Patch-level validation may include same-patient/slide images across development partitions. The source describes non-overlapping training/test patients, but these local file checks cannot independently verify that claim or exclude foundation-model pretraining contamination. No patient identifiers or confidence intervals are invented. Small metric differences do not establish a reliable or universal model ranking. Hardware-dependent timing is not a controlled speed benchmark. Repeated sampling, label efficiency, stain robustness, zero-shot models and whole-slide aggregation are future, separate protocols.

Data: Kather, Halama & Marx (2018), [100,000 histological images of human colorectal cancer and healthy tissue](https://doi.org/10.5281/zenodo.1214456), CC BY 4.0; see [data attribution](docs/data/DATA_ATTRIBUTION.txt). Follow each encoder's model-specific access and licensing terms. Educational research demonstration; no clinical validation.
