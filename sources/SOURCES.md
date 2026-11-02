# Verified sources

Checked 2 November 2026. Dataset metadata and ZIP directory inspection were completed; no model was trained on the dataset during preparation. The Power BI documentation was read, but tenant-specific menu availability, actual browser model creation and DAX execution require the user's signed-in account.

## Data and model

- [Zenodo original collection](https://zenodo.org/records/1214456): provenance, image dimensions/resolution, class descriptions, training normalization and independent evaluation patients.
- [Zenodo metadata export](https://zenodo.org/records/1214456/export/json): creators, v0.1, DOI, CC BY 4.0 licence, filenames, sizes and MD5 checksums. The notebook uses the two verified API content URLs in `dataset_verification.json`.
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): retain attribution and indicate processing when redistributing derived material.
- [torchvision ResNet-18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html): IMAGENET1K_V1 weights and associated preprocessing.
- [scikit-learn 1.7 LogisticRegression](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html): fixed lbfgs classifier; no deprecated multi_class or n_jobs argument is used.
- [scikit-learn F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html): fixed label list and zero-division handling.

## Power BI in a browser

- [Create and edit semantic models in the service](https://learn.microsoft.com/en-us/power-bi/transform-model/service-edit-data-models): current Create/Get data flow, query/model editing, relationships, measures, permissions and tenant limitations.
- [Free-user capabilities](https://learn.microsoft.com/en-us/power-bi/fundamentals/end-user-features): personal My workspace authoring, sharing and public-embed restrictions.
- [CSV files in Power BI](https://learn.microsoft.com/en-us/power-bi/connect-data/service-comma-separated-value-files): upload/cloud-file import options and snapshots.
- [Text/CSV connector](https://learn.microsoft.com/en-us/power-query/connectors/text-csv): host-dependent connector behaviour; a local-path Power Query connection may require a gateway.
- [Web connector](https://learn.microsoft.com/en-us/power-query/connectors/web/web): Web API/Anonymous for public files versus gateway-requiring HTML Web page parsing. Web API as a connector is different from API-key authentication.
- [Web.Contents M function](https://learn.microsoft.com/en-us/powerquery-m/web-contents): HTTPS content and RelativePath option.
- [Data refresh](https://learn.microsoft.com/en-us/power-bi/connect-data/refresh-data): source data refresh versus refreshing visuals; local downloads do not update a cloud source.
- [SELECTEDVALUE](https://learn.microsoft.com/en-us/dax/selectedvalue-function-dax): independent disconnected threshold selection and fallback values.
- [Matrix visual](https://learn.microsoft.com/en-us/power-bi/visuals/power-bi-visualization-matrix-visual): matrix rows, columns and values.
- [Show items with no data](https://learn.microsoft.com/en-us/power-bi/create-reports/desktop-show-items-no-data): complete axes; where available in the browser. The zero-returning measure and separate dimensions provide the core design.
- [Visual interactions](https://learn.microsoft.com/en-us/power-bi/create-reports/service-reports-visual-interactions): disable unintended cross-filtering on the report page.

Unverified details are explicit in the setup: the university's enabled connectors, web-model-editing switch, credentials UI and import replacement controls. A modern HTTPS import is supplied as the gateway-free alternative to host-specific local uploads. Browser authoring cannot bypass a tenant restriction. No clinical validation, runtime estimate or model performance is inferred from these sources.
