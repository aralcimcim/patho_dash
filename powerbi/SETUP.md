# Browser-only Power BI setup

Use https://app.powerbi.com/ on Linux. This guide was checked against Microsoft documentation on **2 November 2026**. I could verify the documented features; I could not inspect your university tenant or execute a report in your signed-in account. Menu rollout and administrator settings can differ.

A **semantic model** is the report's collection of tables, relationships and reusable calculations. A **measure** is a DAX calculation evaluated against the rows selected by the current visual and filters. Power Query imports and types the data; DAX calculates report metrics.

## 1. Account and workspace

Sign in with your university Microsoft account. Open **Workspaces → My workspace**. Personal semantic models and report authoring are supported with a free Power BI licence in My workspace. Collaboration and sharing have different licensing requirements. A Fabric trial is not a prerequisite for this design. [Microsoft: free-user capabilities](https://learn.microsoft.com/en-us/power-bi/fundamentals/end-user-features)

If My workspace or model editing is unavailable, ask the university administrator to check your Power BI access, tenant web-model-editing setting and permissions. You need write permission for the model and Build permission for report authoring; creating your own model normally supplies ownership. Do not buy a licence or start a trial merely because an unrelated Fabric workload advertises one. [Microsoft: web model permissions and tenant settings](https://learn.microsoft.com/en-us/power-bi/transform-model/service-edit-data-models#permissions)

If your tenant blocks all browser model authoring, no Desktop-free workaround can override that restriction. Use the notebook's metrics and confusion figure while the administrator enables access; the full Power BI page requires that access.

## 2. Prepare actual exported data

Run the Colab notebook fully and download its ZIP. Extract it using your browser/file manager. You need these **seven** tables:

| File → table name | Rows | Purpose |
|---|---:|---|
| `predictions.csv` → `predictions` | 7,180 | One held-out patch per row |
| `class_lookup.csv` → `class_lookup` | 9 | True-class dimension |
| `predicted_classes.csv` → `predicted_classes` | 9 | Predicted-class dimension |
| `review_thresholds.csv` → `review_thresholds` | 101 | Independent review control |
| `tumour_thresholds.csv` → `tumour_thresholds` | 101 | Independent tumour control |
| `confidence_bins.csv` → `confidence_bins` | 10 | Histogram intervals |
| `model_metrics.csv` → `model_metrics` | Generated | Fixed Python benchmarks |

`thresholds.csv` is the canonical parameter table; the two named copies avoid sharing a single slicer selection. Keep run metadata and classifier artifacts outside Power BI. Never upload original images or a `.joblib` to the report.

## 3. Choose the browser data source

**Try the modern import first:** select **Create → Get data** (or **New item → Semantic model**, if that opens the same Power Query experience). If your tenant offers an actual CSV **Upload file** without a gateway, upload `predictions.csv`, then add the other six CSVs to the same model through **Get data**. Rename every query exactly as above. Use Transform data to set types, then create the model. A file uploaded this way is a snapshot.

Microsoft documents local CSV uploads, but connector availability differs across hosts. If the Text/CSV dialog asks for an on-premises gateway, cancel and use the HTTPS route below. A path on your computer or Colab is not a cloud-accessible source. [Microsoft: CSV uploads](https://learn.microsoft.com/en-us/power-bi/connect-data/service-comma-separated-value-files), [Text/CSV connector differences](https://learn.microsoft.com/en-us/power-query/connectors/text-csv)

Avoid relying on an older one-CSV quick-import model if it cannot add tables or edit DAX. Create a new model through the modern **Create → Get data** route. [Microsoft: create models in the browser](https://learn.microsoft.com/en-us/power-bi/transform-model/service-edit-data-models#create-new-semantic-models)

### Reliable cloud-source alternative: public HTTPS CSVs

This route needs no local gateway. It makes the selected derived CSVs public. The supplied project has not published anything on your behalf.

1. In GitHub's browser interface create a repository or use one you own. Add a `powerbi-data` folder using **Add file → Upload files**, upload the seven CSVs together and commit. Include the generated `DATA_ATTRIBUTION.txt`. The dataset is CC BY 4.0; retain attribution and describe these as derived research predictions. Do not upload your Drive contents, account data or images.
2. Open `predictions.csv`, select **Raw**, and copy its URL. It should be a direct HTTPS CSV response, such as `https://raw.githubusercontent.com/your-user/your-repository/main/powerbi-data/predictions.csv`, not a GitHub HTML `/blob/` page. Open it in a private browser tab to verify that no login is required.
3. In Power BI choose **Create → Get data → Web API** and use that raw CSV URL, **Anonymous** authentication and no gateway. Here **Web API** is the connector name; do not choose API-key authentication. **Web page** uses a browser engine and may require a gateway. Anonymous `Web.Contents` is suitable for a public CSV. [Microsoft: Power Query Online Web connectors](https://learn.microsoft.com/en-us/power-query/connectors/web/web#connect-from-power-query-online)
4. If credentials/connection creation is unavailable in the editor, open Power BI's **Settings → Manage connections and gateways → New → Cloud**, create the Web/Anonymous cloud connection there, then return to Get data. Names vary by tenant; do not choose on-premises. If the tenant blocks this cloud connector, use university OneDrive/SharePoint through its organisational cloud connector or ask the administrator to allow it. [Microsoft: web-model query limitations](https://learn.microsoft.com/en-us/power-bi/transform-model/service-edit-data-models#limitations)
5. In the query preview, parse as **CSV** if it is shown as binary/text. Choose comma delimiter, UTF-8 and **Use first row as headers** once. Name the query `predictions`. Repeat **Get data → Web API** for the other six direct CSV URLs, with their exact table names. Do not use Web.Page or Web.BrowserContents to parse the CSV.
6. Set the types in section 4, then finish with **Create a semantic model only**, name it `German Pathology AI`, and save in **My workspace**. If the button creates a report instead, save it and open its semantic model to finish relationships/measures. [Microsoft: creation choices](https://learn.microsoft.com/en-us/power-bi/transform-model/service-edit-data-models#create-new-semantic-models)

**Optional faster Power Query path:** if the editor offers **Blank query → Advanced editor**, paste [LoadDashboardCSV.pq](LoadDashboardCSV.pq) as a function query named `LoadDashboardCSV`. Replace `RepoPath` once with your real GitHub user/repository/branch/folder. Create each of the seven queries in [queries.pq](queries.pq); paste one expression per query (start the formula bar with `=`). This applies UTF-8 parsing and all column types automatically. The function is a helper, not an eighth loaded table. If a connection prompt appears, select Anonymous for `https://raw.githubusercontent.com`. Static Web.Contents base URL plus RelativePath supports refresh without embedding credentials. [Microsoft: Web.Contents](https://learn.microsoft.com/en-us/powerquery-m/web-contents), [refresh and dynamic data sources](https://learn.microsoft.com/en-us/power-bi/connect-data/refresh-data#refresh-and-dynamic-data-sources)

Private alternative: use your institution's OneDrive for Business/SharePoint cloud connector and organisational authentication. A private GitHub Raw URL will not work with Anonymous. An HTML sharing page is not a CSV endpoint. Availability of a particular university connector/connection is tenant-dependent; do not publish data publicly if you prefer the private route.

## 4. Data types

In **Transform data**, select columns and use **Data type**. For decimal parsing choose **Using locale → English (United States)** if a German locale interprets decimal points incorrectly. All CSVs have headers and use comma delimiters. Decimal Number means a numeric value, not a percentage string.

| Table | Columns | Power Query / model type |
|---|---|---|
| predictions | image_id, dataset_split, model_name, true_class, predicted_class | Text |
| predictions | confidence and all nine p_* columns | Decimal Number |
| predictions | is_correct, is_tumour_true | Whole Number (0/1) |
| class_lookup, predicted_classes | all columns | Text |
| review_thresholds, tumour_thresholds | threshold | Decimal Number |
| confidence_bins | bin_order | Whole Number |
| confidence_bins | lower, upper | Decimal Number |
| confidence_bins | bin_label | Text |
| model_metrics | metric_name, class_code | Text |
| model_metrics | threshold, value, numerator, denominator | Decimal Number; blanks allowed |

Set image IDs, numeric flags, probabilities and thresholds to **Don't summarize** as their default model summarisation. Use explicit measures for counts and ratios. Do not use “Count of is_correct” as accuracy. Null `model_metrics[value]` means undefined, not zero.

## 5. Relationships

Open **My workspace → semantic model → … → Open semantic model** and switch **Viewing → Editing**. In Model view use **Manage relationships → New**. Create only:

| One side | Many side | Cross-filter direction | Active |
|---|---|---|---|
| class_lookup[class_code] | predictions[true_class] | Single, lookup → predictions | Yes |
| predicted_classes[class_code] | predictions[predicted_class] | Single, lookup → predictions | Yes |

Remove automatically detected relationships involving thresholds, bins or `model_metrics`. Those tables must be disconnected. Never link the two class dimensions to one another or relate one class lookup to both fact columns. That can remove off-diagonal confusion counts. The separate roles are intentional. [Microsoft: relationships in the service](https://learn.microsoft.com/en-us/power-bi/transform-model/service-edit-data-models#create-a-relationship)

If available, select `confidence_bins[bin_label]` and **Sort by column → bin_order**. Otherwise sorting the fixed-width 0.0–0.1 … 0.9–1.0 labels ascending gives the same order. Hide the helper `bin_order` field from the report list if desired.

## 6. Add the complete DAX measures

Select `predictions` in the semantic model Data pane. Choose **New measure** in the ribbon. Paste **one** named expression from [measures.dax](measures.dax), confirm it, then repeat in file order. Measures belong to `predictions` for organisation; they can reference all tables. Do not paste the whole file into one formula box. The `//` lines are comments. [Microsoft: browser measures](https://learn.microsoft.com/en-us/power-bi/transform-model/service-edit-data-models#create-measures)

Counts: whole numbers, no decimals. Accuracy, precision, recall, F1, coverage and proportions: percentage, one or two decimals. Cutoffs: decimal, two places. If your formula editor uses locale-dependent separators, replace argument commas with its expected semicolons, keeping decimal literals valid for that locale.

`DIVIDE(numerator, denominator)` returns blank when the denominator is zero. No retained images means retained accuracy is undefined. No actual TUM images means sensitivity is undefined; no predicted TUM images means tumour precision is undefined. Counts remain zero. Per-class F1 deliberately uses 0 for an absent/unpredicted class so macro F1 averages **all nine classes**, matching Python `labels=CLASSES, zero_division=0`. Macro F1 is blank for an entirely empty cohort.

## 7. Create one report page

From the semantic model choose **Create report → Blank report** (or **New report** in its menu). If a report was created during import, open it and choose **Edit**. Save as `German Pathology AI — Evaluation` in My workspace. Follow [LAYOUT.md](LAYOUT.md) for each visual's exact field wells, filters and interactions. Do not create a separate pinned-tile dashboard; this first version uses one interactive report page.

Two dropdown slicers use `review_thresholds[threshold]` and `tumour_thresholds[threshold]`. Turn **Single select** on and **Select all** off. Select **0.80** and **0.50**. Avoid the default “Between” numeric slicer: a range does not select one parameter row. Defaults in `SELECTEDVALUE` handle no/ambiguous selections but do not make a range control meaningful. [Microsoft: SELECTEDVALUE](https://learn.microsoft.com/en-us/dax/selectedvalue-function-dax)

Filter rules are explicit:

- Counts, accuracy, review metrics and binary metrics obey the current fact-table cohort filters. Class selections may change their denominator if cross-filtering is enabled.
- Per-class precision/recall and macro F1 remove **both class axes** and direct true/predicted class filters when counting class positives; they keep other report filters. Macro F1 always averages nine classes. This avoids computing precision only within a true-class row.
- The two parameter tables filter no images directly. Review confidence changes review/retained measures only. Tumour probability changes only tumour TP/FP/TN/FN measures and ratios; original multiclass argmax stays fixed.
- Imported `model_metrics` benchmarks ignore interactive thresholds and cohort filters. Compare them with DAX only with all cohort filters cleared and the default cutoffs restored.

For the first page, use **Edit interactions** so selecting a matrix cell, class bar, histogram bar or error row does not filter the KPI cards, other charts or controls. Leave parameter-slicer interaction enabled for the relevant measures. See [Microsoft: visual interactions](https://learn.microsoft.com/en-us/power-bi/create-reports/service-reports-visual-interactions). This keeps the page on the complete held-out cohort. The error-only filter belongs to the error visual, never the page.

## 8. Check calculations before taking screenshots

With clear report/page filters and thresholds 0.80/0.50:

1. `[Evaluated Images]` is 7,180; `[Benchmark Evaluated Images]` matches.
2. `[Accuracy Difference]` and `[Macro F1 Difference]` are within 1e-9 of zero. Compare unrounded values using a temporary table visual, not rounded card labels.
3. Matrix grand total is 7,180; its 81 cells match `confusion_matrix.csv` and row sums match evaluation counts in metadata. Off-diagonal cells must remain visible.
4. `[Class Recall]`, `[Class Precision]`, `[Class F1]` per true-class code match the `model_metrics` rows. A class measures table is a useful temporary audit visual.
5. Tumour counts and ratios at 0.50 match Python; `[Tumour Partition Check]` is zero at **every** threshold.
6. Review proportion, coverage and retained accuracy at 0.80 match the fixed benchmarks. Review + retained = evaluated; review proportion + coverage = 1 when the cohort is nonempty.
7. Confidence-bin totals (correct + incorrect) sum to 7,180. Boundary 1.00 belongs to the last bin only. High-confidence errors and rate use **0.90**, independent of the review cutoff.
8. Change only review cutoff: multiclass and tumour benchmark results stay fixed. Change only tumour cutoff: review measures stay fixed. Explore 0, 0.50, 0.80, 0.90, 1.00 to inspect edge cases; do not choose a “best clinical” cutoff from these data.

If a ratio is blank, inspect the corresponding denominator before treating it as a bug. The confidence bins and the lookup roles must not acquire automatic relationships.

## 9. Update a later run

Keep the same baseline to reproduce it. A changed model/protocol is a new experiment, not tuning on this final held-out set.

- **Uploaded snapshots:** downloading another ZIP does not update Power BI. Use the model's supported **Transform data / source replacement** control to upload the replacement CSVs if exposed in your tenant. If it cannot replace an uploaded source, use the HTTPS route for refreshable sources. A new upload can create a different model; do not assume it updates the old report. Keep table names/schema consistent.
- **HTTPS sources:** replace the seven CSVs at the same GitHub URLs in one commit. Keep prior outputs separately by run hash. In the semantic model choose **Refresh → Refresh data only** (or Refresh now from its workspace menu). Check refresh completion, then reopen/refresh report visuals. This fetches the cloud URLs, not a local download or `/content`. Raw-host caches may briefly serve the previous commit; verify row counts/benchmarks before reporting an update. [Microsoft: data refresh](https://learn.microsoft.com/en-us/power-bi/connect-data/refresh-data)
- **Cloud OneDrive/SharePoint sources:** replace files at the connected cloud location and refresh the semantic model. Whether file synchronisation is automatic depends on the specific import/connection type; manual data refresh and checks are sufficient here.

Retain the run metadata and protocol hash alongside your screenshots. If a schema changes, review query types, relationships and DAX rather than silently replacing columns. Do not append repeated predictions for the same single model; replace the snapshot.

## 10. Showcase without a paid requirement

Save the report in My workspace and use screenshots or a browser recording in your portfolio. Free personal authoring does not imply free external sharing, collaboration, or creating a public embed. Microsoft lists Publish to web creation as unavailable to free users; licensing, capacity and administrator controls affect other sharing options. Export availability may also be tenant-controlled. This project therefore requires no public Power BI link. [Microsoft: licensing and export capabilities](https://learn.microsoft.com/en-us/power-bi/fundamentals/end-user-features)
