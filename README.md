# Dashboard

An interactive dashboard for a frozen tissue-classification baseline, evaluated on **7,180 held-out colorectal histology patches**. Explore the confusion matrix, class recall, confidence distribution, individual mistakes, review coverage and tumour probability thresholds.

**[Open the live dashboard](https://aralcimcim.github.io/patho_dash/)**

The website runs entirely in the browser using the saved predictions. It needs no Power BI account, server or model runtime. Changing a threshold recomputes the relevant metrics locally; it does not retrain the model.

## Actual evaluation

| Fixed benchmark                                 |              Result |
| ----------------------------------------------- | ------------------: |
| Held-out patches                                |               7,180 |
| Correct multiclass predictions                  |               6,202 |
| Overall accuracy                                |              86.38% |
| Macro F1, all nine classes                      |              82.77% |
| Tumour sensitivity at p_TUM ≥ 0.50              |              82.40% |
| Tumour specificity at p_TUM ≥ 0.50              |              98.35% |
| Tumour precision at p_TUM ≥ 0.50                |              91.20% |
| Review proportion below confidence 0.80         | 9.93% (713 patches) |
| Retained coverage at confidence ≥ 0.80          |              90.07% |
| Retained accuracy at confidence ≥ 0.80          |              90.69% |
| Errors among predictions with confidence ≥ 0.90 | 460 / 6,120 (7.52%) |

The weakest class recalls are stroma (**46.32%**) and smooth muscle (**70.27%**). The confusion matrix includes 143 stroma → smooth-muscle errors and 124 in the opposite direction. High confidence does not guarantee correctness.

The dashboard verifies the predictions' SHA-256 against [run metadata](docs/data/run_metadata.json), checks the CSV schema and probability consistency, and compares **all 54 fixed metrics** with the saved Python [benchmarks](docs/data/model_metrics.csv) before displaying results. Tests also compare every one of the 81 confusion-matrix cells with the saved matrix.

## Model selection

The dashboard includes a Model selection section showing the current ResNet-18 + logistic regression baseline. The selector stays disabled until additional models are added. The interface uses the portfolio's dark background, neon green accent and system typography.

## Interactions

- **Review confidence cutoff:** flags predictions with maximum class probability strictly below the cutoff. Retained coverage and accuracy use predictions at or above it.
- **Tumour probability threshold:** predicts TUM when `p_TUM >= threshold`. This separate binary decision can differ from multiclass argmax.
- **Matrix, recall and histogram selections:** filter the error table only. Fixed cohort metrics and the other charts remain based on all 7,180 patches.
- **Error table:** search by image ID or tissue code, restrict to high-confidence mistakes, sort confidence, paginate and download the selected rows as CSV.
- **Reset defaults:** restores review 0.80 / tumour 0.50 and clears inspection filters. Thresholds can also be shared in the URL, for example [`?review=0.35&tumour=0.66`](https://aralcimcim.github.io/patho_dash/?review=0.35&tumour=0.66).

Counts display as integers; ratios display as percentages. Undefined ratios display an em dash. Zero-count matrix cells remain visible. The final confidence bin includes exactly 1.0.
