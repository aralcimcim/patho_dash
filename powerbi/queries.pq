// Each expression below is a separate Blank query. Use the exact query name.
// First create the LoadDashboardCSV function from LoadDashboardCSV.pq.

// Query name: predictions
LoadDashboardCSV("predictions.csv")

// Query name: class_lookup
LoadDashboardCSV("class_lookup.csv")

// Query name: predicted_classes
LoadDashboardCSV("predicted_classes.csv")

// Query name: review_thresholds
LoadDashboardCSV("review_thresholds.csv")

// Query name: tumour_thresholds
LoadDashboardCSV("tumour_thresholds.csv")

// Query name: confidence_bins
LoadDashboardCSV("confidence_bins.csv")

// Query name: model_metrics
LoadDashboardCSV("model_metrics.csv")
