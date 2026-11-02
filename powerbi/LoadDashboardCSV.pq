// Power Query M function: create a Blank query, open Advanced editor, paste,
// replace RepoPath with your own PUBLIC CSV folder, and name it LoadDashboardCSV.
// This is a source setting, not a credential. Never put secrets in this function.
(fileName as text) as table =>
let
    RepoPath = "YOUR-USER/YOUR-REPOSITORY/main/powerbi-data",
    Binary = Web.Contents(
        "https://raw.githubusercontent.com",
        [RelativePath = RepoPath & "/" & fileName, Timeout = #duration(0, 0, 3, 0)]
    ),
    CSV = Csv.Document(Binary, [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),
    Headers = Table.PromoteHeaders(CSV, [PromoteAllScalars = true]),
    Types =
        if fileName = "predictions.csv" then {
            {"image_id", type text}, {"dataset_split", type text}, {"model_name", type text},
            {"true_class", type text}, {"predicted_class", type text}, {"confidence", type number},
            {"is_correct", Int64.Type}, {"is_tumour_true", Int64.Type},
            {"p_ADI", type number}, {"p_BACK", type number}, {"p_DEB", type number},
            {"p_LYM", type number}, {"p_MUC", type number}, {"p_MUS", type number},
            {"p_NORM", type number}, {"p_STR", type number}, {"p_TUM", type number}
        }
        else if List.Contains({"class_lookup.csv", "predicted_classes.csv"}, fileName) then {
            {"class_code", type text}, {"english_name", type text}, {"german_name", type text}
        }
        else if List.Contains({"thresholds.csv", "review_thresholds.csv", "tumour_thresholds.csv"}, fileName) then {
            {"threshold", type number}
        }
        else if fileName = "confidence_bins.csv" then {
            {"bin_order", Int64.Type}, {"lower", type number}, {"upper", type number}, {"bin_label", type text}
        }
        else if fileName = "model_metrics.csv" then {
            {"metric_name", type text}, {"class_code", type text}, {"threshold", type number},
            {"value", type number}, {"numerator", type number}, {"denominator", type number}
        }
        else error "Unknown dashboard CSV",
    Typed = Table.TransformColumnTypes(Headers, Types, "en-US")
in
    Typed
