// Synthetic test-only runs derived from the historical CSV. Never publish these files.
import fs from "node:fs";
import { createHash } from "node:crypto";
import {
  CLASSES,
  parseCSV,
  preparePredictions,
  summarize,
  reviewMetrics,
  tumourMetrics,
  toCSV,
} from "../docs/metrics.mjs";
import { extendedMetrics } from "../docs/benchmark.mjs";
const read = (file) =>
  fs.readFileSync(new URL(`../docs/data/${file}`, import.meta.url), "utf8");
const hash = (text) => createHash("sha256").update(text).digest("hex");

function fixedCSV(rows, metadata) {
  const s = summarize(rows),
    t = tumourMetrics(rows, 0.5),
    r = reviewMetrics(rows, 0.8);
  const values = {
    evaluated_images: s.total,
    correct_predictions: s.correct,
    accuracy: s.accuracy,
    macro_f1: s.macroF1,
    tumour_tp: t.tp,
    tumour_fp: t.fp,
    tumour_tn: t.tn,
    tumour_fn: t.fn,
    tumour_sensitivity: t.sensitivity,
    tumour_specificity: t.specificity,
    tumour_precision: t.precision,
    review_images: r.flagged,
    review_proportion: r.reviewProportion,
    coverage: r.coverage,
    retained_accuracy: r.retainedAccuracy,
    high_confidence_images: s.highConfidence,
    high_confidence_errors: s.highConfidenceErrors,
    high_confidence_error_rate: s.highConfidenceErrorRate,
  };
  const records = parseCSV(read("model_metrics.csv")).map((record) => {
    const cls = s.perClass.find((c) => c.code === record.class_code);
    return {
      ...record,
      value: cls
        ? record.metric_name === "support"
          ? cls.actual
          : cls[record.metric_name]
        : values[record.metric_name],
    };
  });
  return toCSV(records, Object.keys(records[0]));
}

export function fixtures() {
  const original = parseCSV(read("predictions.csv"));
  const metadata = JSON.parse(read("run_metadata.json"));
  const models = [],
    files = new Map(),
    runs = [];
  for (let i = 0; i < 2; i++) {
    const id = `test-encoder-${i}`,
      rows = structuredClone(original);
    if (i === 1) {
      const wrong = rows.filter((r) => r.is_correct === "0").slice(0, 2),
        right = rows.find((r) => r.is_correct === "1");
      for (const row of [...wrong, right]) {
        const target =
          row.is_correct === "0"
            ? row.true_class
            : CLASSES.find((c) => c !== row.true_class);
        const former = row.predicted_class;
        [row[`p_${former}`], row[`p_${target}`]] = [
          row[`p_${target}`],
          row[`p_${former}`],
        ];
        row.predicted_class = target;
        row.is_correct = String(Number(target === row.true_class));
      }
      rows.reverse(); // Comparison MUST align by stable ID, not array position.
    }
    for (const row of rows) {
      row.model_name = id;
      row.run_id = id;
    }
    const predictions = toCSV(rows, Object.keys(rows[0]));
    const meta = {
      ...structuredClone(metadata),
      schema_version: 1,
      benchmark_version: "test-only-v1",
      run_id: id,
      predictions_sha256: hash(predictions),
      protocol: {
        ...metadata.protocol,
        model_name: id,
        dataset: { manifest_sha256: "test-only-shared-manifest" },
      },
    };
    const prepared = preparePredictions(parseCSV(predictions), meta);
    const metrics = JSON.stringify(extendedMetrics(prepared));
    meta.metrics_sha256 = hash(metrics);
    const item = {
      id,
      model: id,
      label: `Test encoder ${i === 0 ? "A" : "B"}`,
      status: "available",
      path: `runs/${id}/`,
      benchmark_version: "test-only-v1",
      historical: false,
    };
    models.push(item);
    for (const [file, body] of Object.entries({
      "predictions.csv": predictions,
      "metrics.json": metrics,
      "run_metadata.json": JSON.stringify(meta),
      "model_metrics.csv": fixedCSV(prepared, meta),
    }))
      files.set(`runs/${id}/${file}`, body);
    runs.push({ rows: prepared, metadata: meta, item });
  }
  return {
    files,
    runs,
    registry: { schema_version: 1, default: models[0].id, models },
  };
}
