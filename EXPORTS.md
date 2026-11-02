# Export schemas and validation

No placeholder predictions or invented metric values are supplied. Actual `predictions.csv`, `model_metrics.csv` and `run_metadata.json` are produced when you run the notebook in Colab. Static lookup files can be inspected now in `data/lookups/`.

## Predictions

One row per held-out image, single model, UTF-8 comma-separated CSV, headers:

```text
image_id,dataset_split,model_name,true_class,predicted_class,confidence,is_correct,is_tumour_true,p_ADI,p_BACK,p_DEB,p_LYM,p_MUC,p_MUS,p_NORM,p_STR,p_TUM
```

| Field | Type / rule |
|---|---|
| image_id | Text; archive filename + `/` + original ZIP member path; unique and stable across devices |
| dataset_split | Text; always `held_out_evaluation` |
| model_name | Text; `resnet18_imagenet1k_v1_logreg_c1` |
| true_class, predicted_class | Text; one of the nine fixed codes |
| confidence | Decimal 0–1; maximum probability; uncalibrated |
| is_correct | Integer 0/1; true_class equals predicted_class |
| is_tumour_true | Integer 0/1; true_class is TUM |
| p_ADI through p_TUM | Decimal 0–1; named using the fitted classifier's actual class order |

Expected default evaluated row count: **7,180**, from the complete independent archive. IDs retain both archive and member names rather than basename alone. `class_mapping.json` records the fitted classifier's probability order; the code explicitly indexes that order when assigning named p_* columns.

## Supporting tables

| File | Columns / rows | Notes |
|---|---|---|
| class_lookup.csv | class_code, english_name, german_name; 9 | True-class role; German labels are project translations, not original annotations |
| predicted_classes.csv | Same; 9 | Separate predicted-class role |
| thresholds.csv | threshold; 101 | 0.00 through 1.00 inclusive, step 0.01 |
| review_thresholds.csv | threshold; 101 | Copy for independent review slicer |
| tumour_thresholds.csv | threshold; 101 | Copy for independent tumour slicer |
| confidence_bins.csv | bin_order, lower, upper, bin_label; 10 | [lower, upper), final bin includes 1 |
| model_metrics.csv | metric_name, class_code, threshold, value, numerator, denominator | One fixed benchmark per metric/class; ALL means whole cohort |
| confusion_matrix.csv | true_class plus nine predicted-code columns | Fixed nine-by-nine counts, including zeros |

For metrics, blank threshold means the metric is not thresholded. Blank value means the ratio is undefined. Blank numerator/denominator means those columns do not apply (e.g. macro F1 is an average, not one count ratio). `support` and count metrics are numeric but integral. F1 uses zero for a class absent from both truth and predictions, matching scikit-learn's fixed-label, zero_division=0 macro average.

Fixed benchmark rows cover image count, correct count, accuracy, macro F1, nine-class recall/precision/F1/support, binary tumour confusion counts and sensitivity/specificity/precision at **0.50**, review count/proportion, coverage and retained accuracy at **0.80**, high-confidence count/errors/error rate at **0.90**.

## Reproduction artifacts

- `run_metadata.json`: source links/checksums, CC BY 4.0 citation, frozen protocol and hash, actual versions, counts, class order, model checksum, cache specifications, device, convergence state and evaluation notes. JSON never uses nonstandard NaN.
- `frozen_protocol.json`: baseline configuration and UTC timestamp, written before any held-out inference.
- `train_manifest.csv`, `evaluation_manifest.csv`: stable IDs, archive/member, label, original uncompressed bytes and CRC32; no images. Their file SHA256 values are recorded.
- `grouping_audit.json`: archive counts and the absence of documented grouping metadata.
- `classifier.joblib`: fitted training-only StandardScaler + logistic regression. Load only trusted artifacts and use the recorded environment versions.
- `class_mapping.json`: actual probability order and ImageNet checkpoint SHA256.
- `resolved_versions.txt`, `DATA_ATTRIBUTION.txt`, `confusion_matrix.png`.

Embedding `.npz` caches stay in the cache folder or optional Drive, outside the small deliverables ZIP. The pretrained encoder can be restored with the recorded ResNet18 weights enum and URL; it was frozen and is not a separately fine-tuned model. The ZIP excludes all original images, source ZIPs, embedding caches and pretrained weights.

## Executable checks

Notebook cell 12 defines `validate_predictions` and `metric_table` completely. Cell 14 writes and reloads the CSV, then revalidates it. It checks:

1. Exact schema, nonempty expected row count, and equality to the manifest ID set.
2. Unique, non-null IDs and no missing values in required prediction fields.
3. Single model and evaluation split; valid class codes and actual class order.
4. Finite probabilities in [0, 1], each row summing to 1 (absolute tolerance 1e-8).
5. Argmax in the classifier's actual order agrees with predicted_class, including tie ordering.
6. confidence agrees with maximum probability (absolute tolerance 1e-10).
7. Correctness and tumour flags agree with labels.
8. Confusion matrix contains nine classes and sums to the evaluated image count.
9. Tumour TP+FP+TN+FN sums to evaluated images at the fixed benchmark.
10. Metrics computed from the reloaded CSV agree with the in-memory predictions.

The download cell independently checks original archive sizes and MD5 checksums. The audit cell checks exact per-class archive counts and stops on unexpected files rather than guessing a patient mapping. Cached embeddings are keyed by source/manifest/weights/transform/runtime details and checked for expected ID order, shape and finite values.

The repository's `tests/test_notebook.py` checks these helpers using temporary, clearly synthetic fixtures only. It performs no training on pathology images and contributes no report results. Power BI benchmark comparisons are a separate required check inside the browser; DAX has not been executed in your tenant.
