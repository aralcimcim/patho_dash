import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import {
  CLASSES,
  parseCSV,
  preparePredictions,
  summarize,
  reviewMetrics,
  tumourMetrics,
  checkBenchmarks,
  filterErrors,
  toCSV,
} from "../docs/metrics.mjs";

const read = (file) =>
  readFileSync(new URL(`../docs/data/${file}`, import.meta.url), "utf8");
const metadata = JSON.parse(read("run_metadata.json"));
const records = parseCSV(read("predictions.csv"));
const rows = preparePredictions(records, metadata);
const summary = summarize(rows);
const close = (actual, expected, tolerance = 1e-9) =>
  assert.ok(
    Math.abs(actual - expected) < tolerance,
    `${actual} != ${expected}`,
  );

test("actual prediction export matches the immutable Colab checksum and sample counts", () => {
  assert.equal(
    createHash("sha256").update(read("predictions.csv")).digest("hex"),
    metadata.predictions_sha256,
  );
  assert.equal(rows.length, 7180);
  assert.equal(new Set(rows.map((row) => row.image_id)).size, 7180);
  assert.equal(metadata.train_images, 9000);
});

test("all 54 exported Python benchmarks agree, including every class precision/recall/F1", () => {
  const checks = checkBenchmarks(
    rows,
    parseCSV(read("model_metrics.csv")),
    metadata,
  );
  assert.equal(checks.length, 54);
  assert.ok(
    checks.every((check) => check.matches),
    JSON.stringify(checks.filter((check) => !check.matches)),
  );
});

test("all 81 multiclass confusion cells match the actual saved matrix", () => {
  const expected = parseCSV(read("confusion_matrix.csv"));
  assert.equal(summary.correct, 6202);
  close(summary.accuracy, 6202 / 7180);
  close(summary.macroF1, 0.827719736466);
  assert.deepEqual(
    summary.matrix,
    expected.map((row) => CLASSES.map((code) => Number(row[code]))),
  );
  assert.equal(summary.matrix[7][5], 143);
  assert.equal(summary.matrix[5][7], 124);
});

test("review defaults and the user's Power BI cutoff observations match", () => {
  const review = reviewMetrics(rows, 0.8);
  assert.equal(review.flagged, 713);
  assert.equal(review.retained, 6467);
  close(review.retainedAccuracy, 5865 / 6467);
  assert.equal(reviewMetrics(rows, 0.35).flagged, 3);
  assert.equal(reviewMetrics(rows, 0.54).flagged, 148);
  const all = reviewMetrics(rows, 0);
  assert.equal(all.flagged, 0);
  assert.equal(all.coverage, 1);
  close(all.retainedAccuracy, summary.accuracy);
});

test("tumour metrics reproduce the fixed 0.50 benchmark and the screenshot at 0.66", () => {
  const tumour = tumourMetrics(rows, 0.5);
  assert.deepEqual(
    [tumour.tp, tumour.fp, tumour.tn, tumour.fn],
    [1016, 98, 5849, 217],
  );
  close(tumour.sensitivity, 1016 / 1233);
  close(tumour.specificity, 5849 / 5947);
  assert.equal(
    (tumourMetrics(rows, 0.66).sensitivity * 100).toFixed(2),
    "79.97",
  );
  assert.equal(
    (tumourMetrics(rows, 0.66).specificity * 100).toFixed(2),
    "98.52",
  );
  assert.equal(tumourMetrics(rows, 0).sensitivity, 1);
  assert.equal(tumourMetrics(rows, 0).specificity, 0);
});

test("all 101 cutoffs preserve partitions and monotonic coverage/sensitivity/specificity", () => {
  let previousCoverage = 1,
    previousSensitivity = 1,
    previousSpecificity = 0;
  for (let i = 0; i <= 100; i++) {
    const cutoff = i / 100;
    const review = reviewMetrics(rows, cutoff);
    const tumour = tumourMetrics(rows, cutoff);
    assert.equal(review.flagged + review.retained, 7180);
    close(review.coverage + review.reviewProportion, 1);
    assert.equal(tumour.tp + tumour.fp + tumour.tn + tumour.fn, 7180);
    assert.ok(review.coverage <= previousCoverage);
    assert.ok(tumour.sensitivity <= previousSensitivity);
    assert.ok(tumour.specificity >= previousSpecificity);
    previousCoverage = review.coverage;
    previousSensitivity = tumour.sensitivity;
    previousSpecificity = tumour.specificity;
  }
});

test("confidence distribution partitions every patch once and agrees with high-confidence results", () => {
  assert.equal(
    summary.bins.reduce((sum, bin) => sum + bin.correct + bin.incorrect, 0),
    7180,
  );
  assert.equal(
    summary.bins.reduce((sum, bin) => sum + bin.incorrect, 0),
    978,
  );
  assert.equal(summary.bins[9].incorrect, 460);
  assert.equal(summary.bins[9].correct + summary.bins[9].incorrect, 6120);
  close(summary.highConfidenceErrorRate, 460 / 6120);
});

test("error inspection filters do not change global metrics or mutate predictions", () => {
  assert.equal(filterErrors(rows).length, 978);
  assert.equal(
    filterErrors(rows, { trueClass: "STR", predictedClass: "MUS" }).length,
    143,
  );
  assert.equal(filterErrors(rows, { trueClass: "STR" }).length, 226);
  assert.equal(filterErrors(rows, { highOnly: true }).length, 460);
  assert.equal(filterErrors(rows, { query: "not-a-real-patch" }).length, 0);
  assert.equal(
    filterErrors(rows, { trueClass: "ADI", predictedClass: "ADI" }).length,
    0,
  );
  assert.equal(rows.length, 7180);
  assert.deepEqual(summarize(rows), summary);
});

test("CSV parser and exporter preserve quoted IDs, commas, CRLF, escaped quotes and embedded newlines", () => {
  const fixture =
    '\uFEFFimage_id,true_class\r\n"SYNTHETIC/a,\"\"b\"\"\nline",ADI\r\n';
  const parsed = parseCSV(fixture);
  assert.equal(parsed[0].image_id, 'SYNTHETIC/a,"b"\nline');
  assert.deepEqual(parseCSV(toCSV(parsed, ["image_id", "true_class"])), parsed);
  assert.throws(() => parseCSV("id,id\n1,2"), /Duplicate/);
  assert.throws(() => parseCSV('id\n"unfinished'), /unfinished/);
  assert.throws(() => parseCSV("id,x\n1"), /number of fields/);
});

test("zero denominators, threshold equality and confidence exactly one have explicit semantics", () => {
  // Synthetic arithmetic fixture only, never used as displayed report data.
  const fixture = [
    {
      confidence: 0.5,
      is_correct: 1,
      is_tumour_true: 1,
      p_TUM: 0.5,
      trueIndex: 8,
      predictedIndex: 8,
    },
    {
      confidence: 1,
      is_correct: 0,
      is_tumour_true: 0,
      p_TUM: 1,
      trueIndex: 0,
      predictedIndex: 8,
    },
  ];
  assert.equal(reviewMetrics(fixture, 0.5).flagged, 0);
  assert.equal(reviewMetrics(fixture.slice(0, 1), 1).retainedAccuracy, null);
  assert.equal(tumourMetrics(fixture, 0.5).tp, 1);
  assert.equal(tumourMetrics(fixture, 1).fp, 1);
  assert.equal(tumourMetrics(fixture.slice(0, 1), 1).precision, null);
  assert.equal(tumourMetrics([], 0.5).sensitivity, null);
  assert.equal(summarize([]).macroF1, null);
  assert.equal(summarize(fixture).bins[9].incorrect, 1);
});

test("fixed nine-class macro F1 includes absent classes with zero F1", () => {
  const fixture = [
    { confidence: 1, is_correct: 1, trueIndex: 0, predictedIndex: 0 },
  ];
  close(summarize(fixture).macroF1, 1 / 9);
  assert.equal(summarize(fixture).perClass[1].recall, null);
});

test("validation rejects malformed data rather than silently rendering a different evaluation", () => {
  const smallMetadata = {
    ...metadata,
    evaluated_images: 1,
    evaluation_counts: Object.fromEntries(
      CLASSES.map((c) => [c, c === records[0].true_class ? 1 : 0]),
    ),
  };
  for (const [key, value] of [
    ["p_ADI", "-0.2"],
    ["p_ADI", ""],
    ["confidence", "0"],
    ["is_correct", "3"],
    ["is_correct", ""],
    ["is_tumour_true", ""],
    ["true_class", "UNKNOWN"],
    ["dataset_split", "train"],
    ["model_name", "different_model"],
  ]) {
    assert.throws(() =>
      preparePredictions([{ ...records[0], [key]: value }], smallMetadata),
    );
  }
  assert.throws(() => preparePredictions([...records, records[0]], metadata));
  const benchmarks = parseCSV(read("model_metrics.csv"));
  assert.throws(
    () => checkBenchmarks(rows, benchmarks.slice(1), metadata),
    /incomplete/,
  );
  const wrong = benchmarks.map((row) => ({ ...row }));
  wrong[0].value = "123";
  assert.equal(checkBenchmarks(rows, wrong, metadata)[0].matches, false);
});
