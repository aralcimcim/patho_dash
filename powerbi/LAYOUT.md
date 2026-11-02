# One-page report layout

Create a single report page named **Evaluation**. Use a 1440 × 1080 custom canvas if the browser formatting pane offers custom dimensions; otherwise use 16:9 and adjust proportions. Select the empty canvas → Format page → Canvas settings. [layout.svg](layout.svg) is a wireframe with labels only, not simulated results.

Use a white background, dark navy text (`#19324D`), blue (`#376B97`) for normal quantities, and muted terracotta (`#B85B45`) for errors. Use Segoe UI or another readable sans-serif, 11–12 pt minimum body labels, 20–24 pt title. Keep borders faint and use no gradients, gauges or decorative illustrations. Set ratio axes to 0–100% where meaningful.

## Header and source note

Text box title: **German Pathology AI — Model Evaluation Dashboard**.

Source line: **NCT-CRC-HE-100K → CRC-VAL-HE-7K · Frozen ImageNet ResNet-18 + logistic regression · Patch-level research evaluation**.

Second line: **Source: Kather, Halama & Marx (2018), Zenodo, DOI 10.5281/zenodo.1214456 · CC BY 4.0 · Germany: Heidelberg / Mannheim**. Include the run hash from metadata in a small footer. Use actual training/evaluation counts from that run if adding them to the note; patch count is not patient count.

## Visual placement and field wells

| Position / approximate share | Visual | Fields and settings | What it answers |
|---|---|---|---|
| Top row, four equal cards | Card × 4 | `[Benchmark Evaluated Images]`, `[Benchmark Accuracy]`, `[Benchmark Macro F1]`, `[Tumour Sensitivity]` | Dataset size and headline results |
| Left centre, 50% width | Matrix | Rows `class_lookup[class_code]`; Columns `predicted_classes[class_code]`; Values `[Confusion Count]` | Which true/predicted classes are confused? |
| Right centre, upper | Clustered bar | Y `class_lookup[german_name]`; X `[Class Recall]`; tooltips `[Class Precision]`, `[Class F1]`, `[Class Actual Images]` | Which tissues are recovered reliably? |
| Right centre, lower | Stacked column | X `confidence_bins[bin_label]`; Values `[Confidence Correct]`, `[Confidence Incorrect]` | Where are errors in the confidence distribution? |
| Bottom left | Dropdown slicer + small cards | `review_thresholds[threshold]`; `[Flagged for Review]`, `[Review Proportion]`, `[Coverage]`, `[Retained Accuracy]` | What remains after low-confidence patches are flagged? |
| Bottom right | Dropdown slicer + small cards | `tumour_thresholds[threshold]`; `[Tumour Sensitivity]`, `[Tumour Specificity]`, `[Tumour Precision]` | How does TUM probability threshold change binary decisions? |
| Lower strip | Table | `predictions[image_id]`, `[true_class]`, `[predicted_class]`, `[confidence]`; visual-only filter `is_correct = 0` | Which individual patches were misclassified? |
| Footer / confidence chart subtitle | Small cards/text | `[High Confidence Errors]`, `[High Confidence Error Rate]`; label “confidence ≥ 0.90” | How often are high-confidence predictions wrong? |

Create a visual by selecting its icon, then dragging fields/measures into its field wells in the Build visual/Data pane. To set a visual-only filter, select that visual, drag the field into **Filters on this visual**, and set its value. Enable search or expand the Data pane if a measure is hidden by its table. Save regularly.

Card captions:

- **Evaluated image patches · fixed benchmark**
- **Accuracy · fixed benchmark**
- **Macro F1 · fixed nine classes**
- **TUM sensitivity · exploratory threshold**

First three cards read imported fixed results; the fourth is interactive. Tooltip for tumour sensitivity: **TUM vs other tissue classes; threshold chosen for exploration, not clinical optimisation.** Add the same exploratory label above both control areas. Put `[Review Cutoff]` / `[Tumour Cutoff]` in tooltips or small cards so the active value is unambiguous. Never title the negative class “benign.”

For the histogram rename the two series in this visual to **Correct** and **Incorrect**; use blue and terracotta. Keep bins in ascending order. Intervals are lower-inclusive and upper-exclusive, except 0.9–1.0 includes 1.0. Scores are uncalibrated model confidence, not clinical probability.

For the error table, choose Don't summarize for all columns, format confidence as a percentage, sort descending by confidence, and keep the **is_correct = 0** filter confined to this visual. Let users scroll; 7,180 images do not need to be displayed at once. No image gallery is required.

## Show all confusion cells

Select the matrix. Open the menu beside each class field in Rows/Columns and turn **Show items with no data** on where offered. Use the lookup fields, not the fact class fields, so all nine categories exist. `[Confusion Count]` returns zero for an empty combination. If the menu is absent in your browser build, the separate complete lookup dimensions and zero-returning measure should retain the full grid; check that nine rows and nine columns appear.

Enable grand totals to verify the total is 7,180. Label row axis **True tissue** and column axis **Predicted tissue** using visual titles/subtitles. Apply blue background conditional formatting to Values → Cell elements / Background color (location varies). Counts remain visible; do not replace the counts with row percentages. Keep a linear colour scale and do not suppress off-diagonal zeros. [Microsoft: matrix visual](https://learn.microsoft.com/en-us/power-bi/visuals/power-bi-visualization-matrix-visual), [show items with no data](https://learn.microsoft.com/en-us/power-bi/create-reports/desktop-show-items-no-data)

## Keep filtering understandable

No true/predicted class slicers are required for this first page. Use **Edit interactions** to turn off cross-filtering from matrix, bar chart, histogram and error table to other visuals. Selecting a class should not change the headline cohort unexpectedly. Parameter slicers remain enabled; their disconnected tables affect only measures that read them. A fixed benchmark card never changes with a slicer. A recall bar chart always calculates its class denominator across all predicted classes.

If you later add cohort slicers, clearly label which live measures use the selected cohort. Imported fixed benchmarks continue to describe the full held-out set. Do not compare a subset's live accuracy with the full-set benchmark as though they were the same quantity.

## Screenshots

Save actual report screenshots only after passing SETUP.md's benchmark checks:

- `01_evaluation_overview.png`: full page, default review 0.80 / tumour 0.50.
- `02_confusion_and_class_recall.png`: matrix and per-class recall.
- `03_review_coverage.png`: review control, coverage and retained accuracy; cutoff visible.
- `04_tumour_threshold_exploration.png`: tumour control and sensitivity/specificity; threshold visible and labelled exploratory.

Use a browser screenshot or tenant-enabled PDF/image export. Record the run hash and active thresholds. A screenshot is a valid public portfolio artifact even when an interactive Power BI share needs a paid licence.

Optional later: thumbnails require stable, browser-accessible hosted image URLs and appropriate access/licensing. Colab `/content/...` and local paths cannot serve as Power BI image URLs. Leave images out of this version.
