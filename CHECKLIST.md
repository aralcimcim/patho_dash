# Verification checklist

## Implementation preparation

- [x] Original source, DOI, version, authors and CC BY 4.0 verified.
- [x] Required two archive links, sizes, MD5 values and all class counts verified.
- [x] ZIP directories inspected; no patient/slide mapping found; no filename-derived groups invented.
- [x] Standalone notebook and complete ordered cells supplied without outputs.
- [x] Fixed training-only scaler/classifier and frozen-weight preprocessing implemented.
- [x] Independent parameter tables and class roles supplied; complete DAX and browser guide provided.
- [x] Notebook structure and helper tests pass (19 checks; synthetic fixtures only).

## Published dashboard verification

- [x] Imported the actual 7,180 predictions from the supplied `germ_path` export.
- [x] Prediction and train/evaluation manifest SHA-256 checks agree with metadata.
- [x] All 54 fixed metrics agree with the browser calculations to absolute tolerance 1e-9.
- [x] All 81 confusion-matrix cells match the saved matrix; zeros remain visible.
- [x] All 101 cutoff values preserve count partitions and expected monotonicity.
- [x] Independent controls reproduce the supplied Power BI observations.
- [x] Eight browser tests pass, including layouts at 375, 768 and 1440 pixels.
- [x] Error inspection, download, keyboard use, endpoints and failure states verified.
- [x] Actual browser screenshot and source attribution included.

## Colab execution checklist

The exported metadata records 9,000 training patches, 7,180 evaluation patches and classifier convergence. Artifact checks above were rerun locally. The execution and cache checks below remain a checklist for the person running Colab; they are not a claim of independent runtime observation.

- [ ] Configuration chosen before results; protocol file saved before held-out inference.
- [ ] Required archives pass size/MD5 checks; NONORM was not downloaded.
- [ ] Counts match 100,000 available training / 7,180 evaluation; default sample is 1,000 per class.
- [ ] Only sampled training images and the complete evaluation set extracted.
- [ ] No undocumented patient grouping or random patch validation introduced.
- [ ] Encoder is frozen, eval mode, no gradients; class ordering saved.
- [ ] Scaler and logistic regression fit only the training embeddings.
- [ ] Optimizer convergence checked and recorded; any warning retained in run notes.
- [ ] CSV schema/counts/IDs/probabilities/class mapping pass round-trip validation.
- [ ] Per-class metrics, fixed-nine macro F1 and confusion matrix saved.
- [ ] Fixed tumour 0.50 / review 0.80 / high-confidence 0.90 metrics saved.
- [ ] Actual versions, manifest hashes, source links and evaluation limitations saved.
- [ ] Second embedding-cell execution reuses caches; Drive enabled if restart persistence is desired.
- [ ] ZIP downloaded/backed up; it contains no original images, source archives or embedding caches.

## Optional Power BI report checklist

- [ ] Personal My workspace and browser model editing available in university tenant.
- [ ] Seven tables imported into one semantic model, with exact names/types and UTF-8 labels.
- [ ] Two active single-direction class-role relationships; parameters/bins/metrics disconnected.
- [ ] Every DAX expression added as its own measure; no formula errors.
- [ ] Slicers single-select; defaults review 0.80 / tumour 0.50 visible.
- [ ] Fixed count, accuracy and macro F1 agree with Python benchmarks.
- [ ] All 81 confusion cells visible, including zeros; total 7,180 and all counts match CSV.
- [ ] Per-class recall/precision/F1 agree with Python; macro averages all nine fixed classes.
- [ ] Tumour confusion partition check zero; threshold-0.50 ratios agree with Python.
- [ ] Review metrics at 0.80 agree with Python; review + retained counts sum to evaluated.
- [ ] Confidence bins total evaluated images; exactly 1.0 belongs to final bin once.
- [ ] High-confidence errors/rate use 0.90 and agree with Python.
- [ ] Changing review cutoff does not alter tumour or multiclass results; vice versa for tumour cutoff.
- [ ] Empty-denominator ratios display blank, counts zero; threshold endpoints tested.
- [ ] Error-only filter confined to the error table; visual interactions do not change the full cohort.
- [ ] Fixed benchmarks and exploratory threshold results clearly labelled.
- [ ] No “benign” label for all other tissue classes; no clinical-optimality claim.
- [ ] Screenshot filenames and run hash recorded; actual metrics added to README only after execution.
- [ ] Refresh tested against the actual cloud source/snapshot replacement; no local-path assumptions.
- [ ] Any intentionally public output retains dataset attribution and processing description.
