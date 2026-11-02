// Pure calculations shared by the browser dashboard and Node's benchmark tests.
export const CLASSES = Object.freeze([
  "ADI",
  "BACK",
  "DEB",
  "LYM",
  "MUC",
  "MUS",
  "NORM",
  "STR",
  "TUM",
]);
export const HIGH_CONFIDENCE = 0.9;

export function parseCSV(text) {
  const source = text.replace(/^\uFEFF/, "");
  const records = [];
  let row = [],
    field = "",
    quoted = false;
  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    if (char === '"') {
      if (quoted && source[i + 1] === '"') {
        field += '"';
        i++;
      } else quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(field);
      field = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && source[i + 1] === "\n") i++;
      row.push(field);
      if (row.some((value) => value !== "")) records.push(row);
      row = [];
      field = "";
    } else field += char;
  }
  if (quoted) throw new Error("The CSV contains an unfinished quoted field.");
  if (field !== "" || row.length) {
    row.push(field);
    records.push(row);
  }
  if (!records.length) throw new Error("The CSV is empty.");
  const headers = records.shift();
  if (new Set(headers).size !== headers.length)
    throw new Error("Duplicate CSV column names.");
  return records.map((values, index) => {
    if (values.length !== headers.length)
      throw new Error(`CSV row ${index + 2} has the wrong number of fields.`);
    return Object.fromEntries(headers.map((header, i) => [header, values[i]]));
  });
}

export function ratio(numerator, denominator) {
  return denominator > 0 ? numerator / denominator : null;
}

export function preparePredictions(records, metadata) {
  if (!records.length || records.length !== metadata.evaluated_images)
    throw new Error("Prediction count does not match the saved run.");
  const ids = new Set();
  const order = metadata.classifier_probability_order;
  if (
    !Array.isArray(order) ||
    order.length !== 9 ||
    new Set(order).size !== 9 ||
    order.some((c) => !CLASSES.includes(c))
  ) {
    throw new Error("The classifier class mapping is invalid.");
  }
  const rows = records.map((record, index) => {
    if (!record.image_id || ids.has(record.image_id))
      throw new Error(`Missing or duplicate image ID at row ${index + 1}.`);
    ids.add(record.image_id);
    if (
      !CLASSES.includes(record.true_class) ||
      !CLASSES.includes(record.predicted_class)
    )
      throw new Error("Unknown tissue class.");
    if (
      record.dataset_split !== "held_out_evaluation" ||
      record.model_name !== metadata.protocol.model_name
    ) {
      throw new Error("The CSV contains a different model or dataset split.");
    }
    const probabilities = order.map((code) => {
      const raw = record[`p_${code}`];
      const value = Number(raw);
      if (
        raw === undefined ||
        raw === "" ||
        !Number.isFinite(value) ||
        value < 0 ||
        value > 1
      )
        throw new Error("Invalid class probability.");
      return value;
    });
    if (Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) > 1e-8)
      throw new Error("Class probabilities do not sum to one.");
    const confidence = Math.max(...probabilities);
    const predicted = order[probabilities.indexOf(confidence)];
    if (
      predicted !== record.predicted_class ||
      record.confidence === "" ||
      !Number.isFinite(Number(record.confidence)) ||
      Math.abs(confidence - Number(record.confidence)) > 1e-10
    ) {
      throw new Error(
        "Predicted labels or confidence disagree with probabilities.",
      );
    }
    if (
      !["0", "1"].includes(record.is_correct) ||
      !["0", "1"].includes(record.is_tumour_true)
    ) {
      throw new Error("Correctness and tumour flags must be zero or one.");
    }
    const correct = Number(record.is_correct);
    const tumour = Number(record.is_tumour_true);
    if (
      correct !== Number(record.true_class === predicted) ||
      tumour !== Number(record.true_class === "TUM")
    ) {
      throw new Error(
        "Correctness or tumour flags disagree with class labels.",
      );
    }
    return {
      ...record,
      confidence: Number(record.confidence),
      is_correct: correct,
      is_tumour_true: tumour,
      p_TUM: Number(record.p_TUM),
      trueIndex: CLASSES.indexOf(record.true_class),
      predictedIndex: CLASSES.indexOf(predicted),
    };
  });
  for (const code of CLASSES) {
    if (
      rows.filter((row) => row.true_class === code).length !==
      metadata.evaluation_counts[code]
    )
      throw new Error(`Evaluation count mismatch for ${code}.`);
  }
  return rows;
}

export function summarize(rows) {
  const matrix = CLASSES.map(() => CLASSES.map(() => 0));
  const bins = Array.from({ length: 10 }, (_, i) => ({
    index: i,
    lower: i / 10,
    upper: (i + 1) / 10,
    correct: 0,
    incorrect: 0,
  }));
  let correct = 0,
    highConfidence = 0,
    highConfidenceErrors = 0;
  for (const row of rows) {
    matrix[row.trueIndex][row.predictedIndex]++;
    correct += row.is_correct;
    const bin = bins[Math.min(9, Math.floor(row.confidence * 10))];
    bin[row.is_correct ? "correct" : "incorrect"]++;
    if (row.confidence >= HIGH_CONFIDENCE) {
      highConfidence++;
      highConfidenceErrors += 1 - row.is_correct;
    }
  }
  const perClass = CLASSES.map((code, i) => {
    const actual = matrix[i].reduce((a, b) => a + b, 0);
    const predicted = matrix.reduce((sum, row) => sum + row[i], 0);
    const tp = matrix[i][i];
    return {
      code,
      actual,
      predicted,
      tp,
      recall: ratio(tp, actual),
      precision: ratio(tp, predicted),
      f1: actual + predicted > 0 ? (2 * tp) / (actual + predicted) : 0,
    };
  });
  return {
    total: rows.length,
    correct,
    accuracy: ratio(correct, rows.length),
    matrix,
    perClass,
    bins,
    macroF1: rows.length
      ? perClass.reduce((sum, row) => sum + row.f1, 0) / CLASSES.length
      : null,
    highConfidence,
    highConfidenceErrors,
    highConfidenceErrorRate: ratio(highConfidenceErrors, highConfidence),
  };
}

export function reviewMetrics(rows, cutoff) {
  let retained = 0,
    correctRetained = 0;
  for (const row of rows) {
    if (row.confidence >= cutoff) {
      retained++;
      correctRetained += row.is_correct;
    }
  }
  const flagged = rows.length - retained;
  return {
    cutoff,
    flagged,
    retained,
    correctRetained,
    reviewProportion: ratio(flagged, rows.length),
    coverage: ratio(retained, rows.length),
    retainedAccuracy: ratio(correctRetained, retained),
  };
}

export function tumourMetrics(rows, cutoff) {
  let tp = 0,
    fp = 0,
    tn = 0,
    fn = 0;
  for (const row of rows) {
    const positive = row.p_TUM >= cutoff;
    if (row.is_tumour_true) {
      if (positive) tp++;
      else fn++;
    } else if (positive) fp++;
    else tn++;
  }
  return {
    cutoff,
    tp,
    fp,
    tn,
    fn,
    sensitivity: ratio(tp, tp + fn),
    specificity: ratio(tn, tn + fp),
    precision: ratio(tp, tp + fp),
  };
}

export function checkBenchmarks(rows, benchmarkRows, metadata) {
  const summary = summarize(rows);
  const review = reviewMetrics(rows, metadata.protocol.review_cutoff);
  const tumour = tumourMetrics(rows, metadata.protocol.tumour_threshold);
  const values = {
    evaluated_images: summary.total,
    correct_predictions: summary.correct,
    accuracy: summary.accuracy,
    macro_f1: summary.macroF1,
    tumour_tp: tumour.tp,
    tumour_fp: tumour.fp,
    tumour_tn: tumour.tn,
    tumour_fn: tumour.fn,
    tumour_sensitivity: tumour.sensitivity,
    tumour_specificity: tumour.specificity,
    tumour_precision: tumour.precision,
    review_images: review.flagged,
    review_proportion: review.reviewProportion,
    coverage: review.coverage,
    retained_accuracy: review.retainedAccuracy,
    high_confidence_images: summary.highConfidence,
    high_confidence_errors: summary.highConfidenceErrors,
    high_confidence_error_rate: summary.highConfidenceErrorRate,
  };
  const expectedNames = new Set(
    Object.keys(values).map((name) => `ALL:${name}`),
  );
  for (const cls of summary.perClass)
    for (const name of ["precision", "recall", "f1", "support"])
      expectedNames.add(`${cls.code}:${name}`);
  const seen = new Set();
  const checks = benchmarkRows.map((record) => {
    const key = `${record.class_code}:${record.metric_name}`;
    if (seen.has(key) || !expectedNames.has(key))
      throw new Error(`Unexpected or duplicate benchmark: ${key}`);
    seen.add(key);
    const cls = summary.perClass.find(
      (item) => item.code === record.class_code,
    );
    const actual = cls
      ? record.metric_name === "support"
        ? cls.actual
        : cls[record.metric_name]
      : values[record.metric_name];
    const expected = record.value === "" ? null : Number(record.value);
    const matches =
      expected === null
        ? actual === null
        : actual !== null &&
          Number.isFinite(expected) &&
          Math.abs(actual - expected) < 1e-9;
    return {
      name: record.metric_name,
      classCode: record.class_code,
      expected,
      actual,
      matches,
    };
  });
  if (seen.size !== expectedNames.size)
    throw new Error("The fixed benchmark file is incomplete.");
  return checks;
}

export function filterErrors(
  rows,
  {
    trueClass = "",
    predictedClass = "",
    bin = null,
    query = "",
    highOnly = false,
  } = {},
) {
  const search = query.trim().toLowerCase();
  return rows.filter(
    (row) =>
      !row.is_correct &&
      (!trueClass || row.true_class === trueClass) &&
      (!predictedClass || row.predicted_class === predictedClass) &&
      (bin === null ||
        Math.min(9, Math.floor(row.confidence * 10)) === Number(bin)) &&
      (!highOnly || row.confidence >= HIGH_CONFIDENCE) &&
      (!search ||
        row.image_id.toLowerCase().includes(search) ||
        row.true_class.toLowerCase().includes(search) ||
        row.predicted_class.toLowerCase().includes(search)),
  );
}

export function toCSV(rows, columns) {
  const quote = (value) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  return (
    [
      columns.join(","),
      ...rows.map((row) => columns.map((key) => quote(row[key])).join(",")),
    ].join("\r\n") + "\r\n"
  );
}
