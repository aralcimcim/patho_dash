import {
  CLASSES,
  summarize,
  tumourMetrics,
  reviewMetrics,
} from "./metrics.mjs";

export function compareModels(first, second) {
  const a = first.metadata,
    b = second.metadata;
  if (
    !a.benchmark_version ||
    a.benchmark_version !== b.benchmark_version ||
    !a.protocol.dataset?.manifest_sha256 ||
    a.protocol.dataset.manifest_sha256 !== b.protocol.dataset?.manifest_sha256
  )
    throw new Error(
      "Comparison requires the same benchmark version and exact shared manifest.",
    );
  const right = new Map(second.rows.map((row) => [row.image_id, row]));
  if (
    right.size !== second.rows.length ||
    first.rows.length !== right.size ||
    new Set(first.rows.map((r) => r.image_id)).size !== right.size
  )
    throw new Error("Models do not contain the same unique image IDs.");
  const cases = first.rows.map((left) => {
    const row = right.get(left.image_id);
    if (!row || row.true_class !== left.true_class)
      throw new Error("Model image IDs or true labels do not align.");
    return {
      image_id: left.image_id,
      true_class: left.true_class,
      first_prediction: left.predicted_class,
      second_prediction: row.predicted_class,
      corrected: !left.is_correct && !!row.is_correct,
      regression: !!left.is_correct && !row.is_correct,
      both_wrong: !left.is_correct && !row.is_correct,
      disagreement: left.predicted_class !== row.predicted_class,
    };
  });
  const before = summarize(first.rows),
    after = summarize(second.rows);
  return {
    cases,
    recall: CLASSES.map((code, i) => ({
      code,
      first: before.perClass[i].recall,
      second: after.perClass[i].recall,
      difference: after.perClass[i].recall - before.perClass[i].recall,
    })),
    counts: Object.fromEntries(
      ["corrected", "regression", "both_wrong", "disagreement"].map((key) => [
        key,
        cases.filter((r) => r[key]).length,
      ]),
    ),
  };
}

function rankingMetrics(labels, scores) {
  const positives = labels.filter(Boolean).length,
    negatives = labels.length - positives;
  if (!positives || !negatives) return { auc: null, ap: positives ? 1 : null };
  const rows = scores
    .map((score, i) => ({ score, positive: labels[i] }))
    .sort((a, b) => b.score - a.score);
  let tp = 0,
    fp = 0,
    area = 0,
    ap = 0;
  for (let i = 0; i < rows.length;) {
    let j = i,
      added = 0;
    while (j < rows.length && rows[j].score === rows[i].score) {
      added += Number(rows[j].positive);
      j++;
    }
    const previousTP = tp,
      previousFP = fp;
    tp += added;
    fp += j - i - added;
    area +=
      (((fp - previousFP) / negatives) * (tp + previousTP)) / (2 * positives);
    ap += ((added / positives) * tp) / (tp + fp);
    i = j;
  }
  return { auc: area, ap };
}

export function extendedMetrics(rows) {
  const s = summarize(rows),
    t = tumourMetrics(rows, 0.5);
  const aucs = CLASSES.map(
    (code) =>
      rankingMetrics(
        rows.map((r) => r.true_class === code),
        rows.map((r) => Number(r[`p_${code}`])),
      ).auc,
  );
  const tumour = rankingMetrics(
    rows.map((r) => !!r.is_tumour_true),
    rows.map((r) => r.p_TUM),
  );
  let loss = 0,
    brier = 0;
  for (const row of rows) {
    const p = CLASSES.map((c) => Number(row[`p_${c}`]));
    const clipped = p.map((v) =>
      Math.max(Number.EPSILON, Math.min(1 - Number.EPSILON, v)),
    );
    const normalization = clipped.reduce((a, b) => a + b, 0);
    loss -= Math.log(clipped[row.trueIndex] / normalization);
    brier += p.reduce(
      (sum, v, i) => sum + (v - Number(i === row.trueIndex)) ** 2,
      0,
    );
  }
  return {
    accuracy: s.accuracy,
    balanced_accuracy: s.perClass.reduce((n, c) => n + c.recall, 0) / 9,
    macro_f1: s.macroF1,
    macro_auroc: aucs.every((v) => v !== null)
      ? aucs.reduce((a, b) => a + b, 0) / 9
      : null,
    log_loss: loss / rows.length,
    brier_score: brier / rows.length,
    per_class: s.perClass.map((c, i) => ({
      class_code: c.code,
      precision: c.precision,
      recall: c.recall,
      f1: c.f1,
      support: c.actual,
      auroc: aucs[i],
    })),
    confusion_matrix: s.matrix,
    confusion_matrix_row_normalized: s.matrix.map((r, i) =>
      r.map((n) => n / s.perClass[i].actual),
    ),
    tumour: {
      threshold: 0.5,
      sensitivity: t.sensitivity,
      specificity: t.specificity,
      precision: t.precision,
      f1: (2 * t.tp) / (2 * t.tp + t.fp + t.fn),
      tp: t.tp,
      fp: t.fp,
      tn: t.tn,
      fn: t.fn,
      auroc: tumour.auc,
      average_precision: tumour.ap,
      confusion_matrix: [
        [t.tn, t.fp],
        [t.fn, t.tp],
      ],
    },
    review_curve: Array.from({ length: 101 }, (_, i) => {
      const r = reviewMetrics(rows, i / 100);
      return {
        cutoff: i / 100,
        coverage: r.coverage,
        retained_accuracy: r.retainedAccuracy,
        retained_error:
          r.retainedAccuracy === null ? null : 1 - r.retainedAccuracy,
      };
    }),
  };
}

export function validateExtended(rows, saved) {
  function visit(actual, expected, path) {
    if (actual && typeof actual === "object") {
      if (!expected || typeof expected !== "object")
        throw new Error(`Missing saved metric: ${path}`);
      for (const key of Object.keys(actual))
        visit(actual[key], expected[key], `${path}.${key}`);
    } else if (typeof actual === "number") {
      if (
        typeof expected !== "number" ||
        !Number.isFinite(expected) ||
        Math.abs(actual - expected) > 1e-9
      )
        throw new Error(`Saved metric disagrees with predictions: ${path}`);
    } else if (actual !== expected)
      throw new Error(`Saved metric disagrees with predictions: ${path}`);
  }
  visit(extendedMetrics(rows), saved, "metrics");
}
