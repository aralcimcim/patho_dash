import {
  compareModels,
  extendedMetrics,
  validateExtended,
} from "./benchmark.mjs";
import {
  CLASSES,
  HIGH_CONFIDENCE,
  parseCSV,
  preparePredictions,
  summarize,
  reviewMetrics,
  tumourMetrics,
  checkBenchmarks,
  filterErrors,
  toCSV,
} from "./metrics.mjs";

const $ = (id) => document.getElementById(id);
const count = (value) => new Intl.NumberFormat("en-GB").format(value);
const percent = (value) =>
  value === null
    ? "—"
    : new Intl.NumberFormat("en-GB", {
        style: "percent",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(value);
const setText = (id, value) => {
  $(id).textContent = value;
};
const PAGE_SIZE = 20;
const state = {
  rows: [],
  lookup: new Map(),
  summary: null,
  metadata: null,
  review: 0.8,
  tumour: 0.5,
  trueClass: "",
  predictedClass: "",
  bin: null,
  query: "",
  highOnly: false,
  page: 0,
  descending: true,
};
const modelRuns = new Map();
let registry;
let comparison;
let leaderboardSort = { key: "macro_f1", descending: true };

const exportColumns = [
  "image_id",
  "dataset_split",
  "model_name",
  "true_class",
  "predicted_class",
  "confidence",
  "is_correct",
  "is_tumour_true",
  ...CLASSES.map((c) => `p_${c}`),
  "run_id",
];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function readCutoff(params, name, fallback) {
  const raw = params.get(name);
  const value = Number(raw);
  return raw !== null &&
    raw.trim() !== "" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
    ? Math.round(value * 100) / 100
    : fallback;
}

function updateURL() {
  const url = new URL(location.href);
  url.searchParams.set("review", state.review.toFixed(2));
  url.searchParams.set("tumour", state.tumour.toFixed(2));
  history.replaceState(null, "", url);
}

function renderSummary() {
  const s = state.summary;
  setText("evaluated-images", count(s.total));
  setText("overall-accuracy", percent(s.accuracy));
  setText("macro-f1", percent(s.macroF1));
  setText("high-confidence-errors", count(s.highConfidenceErrors));
  setText("high-confidence-rate", percent(s.highConfidenceErrorRate));
  setText(
    "high-confidence-denominator",
    `${count(s.highConfidenceErrors)} errors / ${count(s.highConfidence)} predictions ≥ 0.90`,
  );
}

function renderThresholds() {
  for (const name of ["review", "tumour"]) {
    $(`${name}-range`).value = Math.round(state[name] * 100);
    $(`${name}-range`).setAttribute("aria-valuetext", state[name].toFixed(2));
    $(`${name}-value`).value = state[name].toFixed(2);
  }
  const review = reviewMetrics(state.rows, state.review);
  setText("review-flagged", count(review.flagged));
  setText("review-proportion", percent(review.reviewProportion));
  setText("review-coverage", percent(review.coverage));
  setText("retained-accuracy", percent(review.retainedAccuracy));
  setText(
    "review-count-note",
    `${count(review.retained)} retained patches; ${count(review.correctRetained)} correctly classified. Multiclass predictions stay fixed.`,
  );
  const tumour = tumourMetrics(state.rows, state.tumour);
  setText("tumour-sensitivity", percent(tumour.sensitivity));
  setText("tumour-specificity", percent(tumour.specificity));
  setText("tumour-precision", percent(tumour.precision));
  setText(
    "tumour-count-note",
    `TP ${count(tumour.tp)} · FP ${count(tumour.fp)} · TN ${count(tumour.tn)} · FN ${count(tumour.fn)}. Negative category: other tissue classes.`,
  );
}

function renderMatrix() {
  $("confusion-matrix").replaceChildren();
  const normalized = $("matrix-mode").value === "normalized";
  const { matrix, perClass, total } = state.summary;
  const head = element("thead");
  const heading = element("tr");
  const axis = element("th", "matrix-axis", "True / predicted");
  axis.scope = "col";
  heading.append(axis);
  for (const code of [...CLASSES, "Total"]) {
    const th = element("th", "", code);
    th.scope = "col";
    if (state.lookup.has(code)) th.title = state.lookup.get(code).english_name;
    heading.append(th);
  }
  head.append(heading);
  const body = element("tbody");
  const maximum = Math.max(...matrix.flat());
  const palette = getComputedStyle(document.documentElement);
  const rgbToken = (name) =>
    palette
      .getPropertyValue(name)
      .trim()
      .slice(1)
      .match(/.{2}/g)
      .map((value) => parseInt(value, 16));
  const accent = rgbToken("--accent");
  const surface = rgbToken("--surface");
  for (let i = 0; i < CLASSES.length; i++) {
    const tr = element("tr");
    const label = element("th", "", CLASSES[i]);
    label.scope = "row";
    label.title = state.lookup.get(CLASSES[i]).english_name;
    tr.append(label);
    for (let j = 0; j < CLASSES.length; j++) {
      const n = matrix[i][j];
      const td = element("td", i === j ? "diagonal" : "");
      const button = element(
        "button",
        "matrix-count",
        normalized ? percent(n / perClass[i].actual) : count(n),
      );
      button.type = "button";
      button.dataset.trueClass = CLASSES[i];
      button.dataset.predictedClass = CLASSES[j];
      button.setAttribute(
        "aria-label",
        `True ${CLASSES[i]}, predicted ${CLASSES[j]}: ${count(n)} patches${i === j ? ", correct predictions" : ", inspect errors"}`,
      );
      button.title = `${CLASSES[i]} → ${CLASSES[j]}: ${count(n)} patches`;
      const strength = normalized
        ? n / perClass[i].actual
        : maximum
          ? n / maximum
          : 0;
      const opacity = n ? 0.08 + 0.8 * strength : 0;
      const rgb = surface.map((base, index) =>
        Math.round(base + (accent[index] - base) * opacity),
      );
      button.style.backgroundColor = `rgb(${rgb.join(", ")})`;
      // Choose a readable label for each shade of the shared green palette.
      const luminance = rgb
        .map((value) => value / 255)
        .map((value) =>
          value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
        )
        .reduce(
          (sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index],
          0,
        );
      button.style.color = luminance > 0.179 ? "#091007" : "#fff";
      if (i === j || n === 0) button.disabled = true;
      else {
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("click", () => {
          const repeated =
            state.trueClass === CLASSES[i] &&
            state.predictedClass === CLASSES[j];
          state.trueClass = repeated ? "" : CLASSES[i];
          state.predictedClass = repeated ? "" : CLASSES[j];
          state.page = 0;
          renderSelection();
          renderErrors();
        });
      }
      td.append(button);
      tr.append(td);
    }
    tr.append(element("td", "total", count(perClass[i].actual)));
    body.append(tr);
  }
  const footer = element("tfoot");
  const totals = element("tr");
  const totalLabel = element("th", "", "Total");
  totalLabel.scope = "row";
  totals.append(totalLabel);
  for (const cls of perClass)
    totals.append(element("td", "total", count(cls.predicted)));
  totals.append(element("td", "total", count(total)));
  footer.append(totals);
  $("confusion-matrix").append(head, body, footer);
}

function renderRecall() {
  $("recall-chart").replaceChildren();
  const sorted = [...state.summary.perClass].sort(
    (a, b) => (b.recall ?? -1) - (a.recall ?? -1),
  );
  for (const cls of sorted) {
    const names = state.lookup.get(cls.code);
    const button = element("button", "recall-row");
    button.type = "button";
    button.dataset.classCode = cls.code;
    button.setAttribute("aria-pressed", "false");
    const description = `${names.english_name} · ${names.german_name}. Recall ${percent(cls.recall)}, precision ${percent(cls.precision)}, F1 ${percent(cls.f1)}; ${count(cls.actual)} actual patches.`;
    button.title = description;
    button.setAttribute(
      "aria-label",
      `${cls.code}: ${description} Inspect errors.`,
    );
    const track = element("span", "recall-track");
    const bar = element("span", "recall-bar");
    bar.style.width = `${(cls.recall ?? 0) * 100}%`;
    track.append(bar);
    button.append(
      element("span", "recall-code", cls.code),
      track,
      element("span", "recall-value", percent(cls.recall)),
    );
    button.addEventListener("click", () => {
      const repeated = state.trueClass === cls.code && !state.predictedClass;
      state.trueClass = repeated ? "" : cls.code;
      state.predictedClass = "";
      state.page = 0;
      renderSelection();
      renderErrors();
    });
    $("recall-chart").append(button);
  }
}

function renderConfidence() {
  $("confidence-chart").replaceChildren();
  const maximum = Math.max(
    ...state.summary.bins.map((bin) => bin.correct + bin.incorrect),
  );
  for (const bin of state.summary.bins) {
    const total = bin.correct + bin.incorrect;
    const label = `${bin.lower.toFixed(1)}–${bin.upper.toFixed(1)}`;
    const button = element("button", "confidence-bin");
    button.type = "button";
    button.dataset.bin = bin.index;
    button.setAttribute("aria-pressed", "false");
    button.setAttribute(
      "aria-label",
      `Confidence ${label}: ${count(bin.correct)} correct, ${count(bin.incorrect)} incorrect. Inspect errors.`,
    );
    button.title = `${label}: ${count(total)} patches (${count(bin.correct)} correct, ${count(bin.incorrect)} incorrect)`;
    const plot = element("span", "bin-plot");
    const stack = element("span", "bin-stack");
    stack.style.height = `${maximum ? (100 * total) / maximum : 0}%`;
    const error = element("span", "bin-incorrect");
    error.style.height = `${total ? (100 * bin.incorrect) / total : 0}%`;
    const correct = element("span", "bin-correct");
    correct.style.height = `${total ? (100 * bin.correct) / total : 0}%`;
    stack.append(error, correct);
    plot.append(stack);
    button.append(
      element("span", "bin-total", count(total)),
      plot,
      element("span", "bin-label", label),
    );
    button.addEventListener("click", () => {
      state.bin = state.bin === bin.index ? null : bin.index;
      state.page = 0;
      renderSelection();
      renderErrors();
    });
    $("confidence-chart").append(button);
  }
}

function renderSelection() {
  document.querySelectorAll(".matrix-count[aria-pressed]").forEach((button) => {
    button.setAttribute(
      "aria-pressed",
      String(
        button.dataset.trueClass === state.trueClass &&
          button.dataset.predictedClass === state.predictedClass,
      ),
    );
  });
  document
    .querySelectorAll(".recall-row")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.classCode === state.trueClass),
      ),
    );
  document
    .querySelectorAll(".confidence-bin")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(Number(button.dataset.bin) === state.bin),
      ),
    );
}

function selectedErrors() {
  return filterErrors(state.rows, state).sort((a, b) => {
    const difference = state.descending
      ? b.confidence - a.confidence
      : a.confidence - b.confidence;
    return difference || a.image_id.localeCompare(b.image_id);
  });
}

function renderErrors() {
  const errors = selectedErrors();
  const pages = Math.max(1, Math.ceil(errors.length / PAGE_SIZE));
  state.page = Math.min(state.page, pages - 1);
  const body = $("error-rows");
  body.replaceChildren();
  for (const row of errors.slice(
    state.page * PAGE_SIZE,
    (state.page + 1) * PAGE_SIZE,
  )) {
    const tr = element("tr");
    const idCell = element("td");
    const code = element("code", "image-id", row.image_id.split("/").at(-1));
    code.title = row.image_id;
    idCell.append(code);
    tr.append(idCell);
    for (const value of [row.true_class, row.predicted_class])
      tr.append(element("td", "", value));
    const confidence = element("td", "numeric", row.confidence.toFixed(6));
    confidence.title = row.confidence.toPrecision(14);
    const tumour = element("td", "numeric", row.p_TUM.toFixed(6));
    tumour.title = row.p_TUM.toPrecision(14);
    tr.append(confidence, tumour);
    body.append(tr);
  }
  if (!errors.length) {
    const td = element(
      "td",
      "empty-table",
      "No misclassified patches match these inspection filters.",
    );
    td.colSpan = 5;
    const tr = element("tr");
    tr.append(td);
    body.append(tr);
  }
  const filters = [];
  if (state.trueClass) filters.push(`true ${state.trueClass}`);
  if (state.predictedClass) filters.push(`predicted ${state.predictedClass}`);
  if (state.bin !== null)
    filters.push(
      `confidence ${(state.bin / 10).toFixed(1)}–${((state.bin + 1) / 10).toFixed(1)}`,
    );
  if (state.highOnly) filters.push("confidence ≥ 0.90");
  if (state.query) filters.push("text search active");
  setText(
    "error-scope",
    `${count(errors.length)} of ${count(state.summary.total - state.summary.correct)} errors${filters.length ? ` · ${filters.join(" · ")}` : " · All errors"}. Image IDs are shortened to filenames; hover for archive paths.`,
  );
  setText("error-page-status", `Page ${state.page + 1} of ${pages}`);
  $("previous-page").disabled = state.page === 0;
  $("next-page").disabled = state.page >= pages - 1;
  $("download-errors").disabled = !errors.length;
  $("confidence-sort-header").setAttribute(
    "aria-sort",
    state.descending ? "descending" : "ascending",
  );
  setText("sort-arrow", state.descending ? "↓" : "↑");
}

function clearErrorFilters() {
  state.trueClass = "";
  state.predictedClass = "";
  state.bin = null;
  state.query = "";
  state.highOnly = false;
  state.page = 0;
  $("error-search").value = "";
  $("high-errors-only").checked = false;
  renderSelection();
  renderErrors();
}

function bindControls() {
  for (const name of ["review", "tumour"]) {
    $(`${name}-range`).addEventListener("input", (event) => {
      state[name] = Number(event.target.value) / 100;
      renderThresholds();
      updateURL();
    });
    $(`${name}-value`).addEventListener("input", (event) => {
      const value = event.target.valueAsNumber;
      if (
        event.target.value === "" ||
        !Number.isFinite(value) ||
        value < 0 ||
        value > 1
      )
        return;
      state[name] = Math.round(value * 100) / 100;
      const input = event.target;
      // Keep the number field editable while typing; reformat on change/blur.
      const entered = input.value;
      renderThresholds();
      input.value = entered;
      updateURL();
    });
    $(`${name}-value`).addEventListener("change", () => renderThresholds());
    $(`${name}-value`).addEventListener("blur", () => renderThresholds());
  }
  $("reset-defaults").disabled = false;
  $("reset-defaults").addEventListener("click", () => {
    state.review = state.metadata.protocol.review_cutoff;
    state.tumour = state.metadata.protocol.tumour_threshold;
    state.descending = true;
    clearErrorFilters();
    renderThresholds();
    updateURL();
  });
  $("clear-errors").addEventListener("click", clearErrorFilters);
  $("error-search").addEventListener("input", (event) => {
    state.query = event.target.value;
    state.page = 0;
    renderErrors();
  });
  $("high-errors-only").addEventListener("change", (event) => {
    state.highOnly = event.target.checked;
    state.page = 0;
    renderErrors();
  });
  $("previous-page").addEventListener("click", () => {
    state.page--;
    renderErrors();
  });
  $("next-page").addEventListener("click", () => {
    state.page++;
    renderErrors();
  });
  $("sort-confidence").addEventListener("click", () => {
    state.descending = !state.descending;
    state.page = 0;
    renderErrors();
  });
  $("download-errors").addEventListener("click", () => {
    const blob = new Blob([toCSV(selectedErrors(), exportColumns)], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = element("a");
    link.href = url;
    link.download = "pathology-inspected-errors.csv";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

async function fetchText(file) {
  const response = await fetch(new URL(`./data/${file}`, import.meta.url));
  if (!response.ok)
    throw new Error(`Could not load ${file} (HTTP ${response.status}).`);
  return response.text();
}

async function sha256(text) {
  if (!globalThis.crypto?.subtle)
    throw new Error("Secure context required to verify exported results.");
  const bytes = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(text),
  );
  return [...new Uint8Array(bytes)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function loadModel(item) {
  if (!/^(runs\/[a-z0-9-]+\/)?$/.test(item.path))
    throw new Error("Invalid model artifact path.");
  const [predictionsText, metricsText, metadataText] = await Promise.all([
    fetchText(item.path + "predictions.csv"),
    fetchText(item.path + "model_metrics.csv"),
    fetchText(item.path + "run_metadata.json"),
  ]);
  const metadata = JSON.parse(metadataText);
  if ((await sha256(predictionsText)) !== metadata.predictions_sha256)
    throw new Error("Prediction checksum differs from the saved run metadata.");
  const rows = preparePredictions(parseCSV(predictionsText), metadata);
  const checks = checkBenchmarks(rows, parseCSV(metricsText), metadata);
  if (checks.some((c) => !c.matches))
    throw new Error(
      "Browser calculations disagree with the fixed Python benchmarks.",
    );
  if (
    !item.historical &&
    (metadata.schema_version !== 1 ||
      metadata.run_id !== item.id ||
      metadata.benchmark_version !== item.benchmark_version)
  )
    throw new Error("Model registry and run metadata disagree.");
  let metrics = extendedMetrics(rows);
  if (!item.historical) {
    const saved = await fetchText(item.path + "metrics.json");
    if ((await sha256(saved)) !== metadata.metrics_sha256)
      throw new Error("Aggregate metric checksum differs.");
    metrics = JSON.parse(saved);
    validateExtended(rows, metrics);
  }
  return { item, rows, metadata, metrics, summary: summarize(rows) };
}

function renderLeaderboard() {
  const tbody = $("leaderboard-rows");
  tbody.replaceChildren();
  const values = (item) => {
    const run = modelRuns.get(item.id),
      m = run?.metrics,
      e = run?.metadata.extraction;
    return {
      label: item.label,
      macro_f1: m?.macro_f1,
      accuracy: m?.accuracy,
      sensitivity: m?.tumour.sensitivity,
      specificity: m?.tumour.specificity,
      seconds: e?.seconds,
      memory: e?.peak_gpu_memory_gb,
    };
  };
  const items = [...registry.models].sort((a, b) => {
    const av = values(a)[leaderboardSort.key],
      bv = values(b)[leaderboardSort.key];
    if (av == null) return bv == null ? 0 : 1;
    if (bv == null) return -1;
    return (
      (typeof av === "string" ? av.localeCompare(bv) : av - bv) *
      (leaderboardSort.descending ? -1 : 1)
    );
  });
  for (const item of items) {
    const v = values(item),
      tr = element("tr");
    tr.append(element("td", "", item.label));
    for (const key of ["macro_f1", "accuracy", "sensitivity", "specificity"])
      tr.append(element("td", "", percent(v[key] ?? null)));
    tr.append(
      element("td", "", v.seconds == null ? "—" : v.seconds.toFixed(1)),
      element("td", "", v.memory == null ? "—" : v.memory.toFixed(2)),
    );
    tr.append(
      element(
        "td",
        "",
        `${item.benchmark_version} · ${modelRuns.has(item.id) ? (item.historical ? "Historical reference" : "Complete") : item.reason || "Unavailable"}`,
      ),
    );
    tbody.append(tr);
  }
  document.querySelectorAll("[data-sort]").forEach((button) => {
    const active = button.dataset.sort === leaderboardSort.key;
    button.parentElement.setAttribute(
      "aria-sort",
      active
        ? leaderboardSort.descending
          ? "descending"
          : "ascending"
        : "none",
    );
    button.textContent =
      button.textContent.replace(/ [↓↑]$/, "") +
      (active ? (leaderboardSort.descending ? " ↓" : " ↑") : "");
  });
}

function renderComparison() {
  $("comparison-results").hidden = true;
  comparison = null;
  const first = modelRuns.get($("compare-first").value),
    second = modelRuns.get($("compare-second").value);
  if (!first || !second || first === second) {
    setText(
      "comparison-status",
      "Publish two completed runs under the same benchmark protocol to compare them.",
    );
    return;
  }
  try {
    comparison = compareModels(first, second);
  } catch (error) {
    setText("comparison-status", error.message);
    return;
  }
  setText(
    "comparison-status",
    `${first.item.label} → ${second.item.label} · ${count(first.rows.length)} aligned test patches`,
  );
  $("comparison-recall").replaceChildren();
  for (const c of comparison.recall) {
    const tr = element("tr");
    for (const v of [
      c.code,
      percent(c.first),
      percent(c.second),
      `${c.difference >= 0 ? "+" : ""}${(100 * c.difference).toFixed(2)}`,
    ])
      tr.append(element("td", "", v));
    $("comparison-recall").append(tr);
  }
  $("comparison-results").hidden = false;
  renderComparisonCases();
}

function comparisonCases() {
  return (
    comparison?.cases.filter((row) => row[$("comparison-category").value]) || []
  );
}
function renderComparisonCases() {
  if (!comparison) return;
  const c = comparison.counts;
  setText(
    "comparison-counts",
    `Corrected ${count(c.corrected)} · New errors ${count(c.regression)} · Both wrong ${count(c.both_wrong)} · Disagreements ${count(c.disagreement)}`,
  );
  $("comparison-cases").replaceChildren();
  for (const row of comparisonCases().slice(0, 20)) {
    const tr = element("tr");
    for (const key of [
      "image_id",
      "true_class",
      "first_prediction",
      "second_prediction",
    ]) {
      const td = element(
        "td",
        "",
        key === "image_id" ? row[key].split("/").at(-1) : row[key],
      );
      td.title = row[key];
      tr.append(td);
    }
    $("comparison-cases").append(tr);
  }
  $("download-comparison").disabled = !comparisonCases().length;
}

function downloadCSV(rows, columns, filename) {
  const url = URL.createObjectURL(
    new Blob([toCSV(rows, columns)], { type: "text/csv;charset=utf-8" }),
  );
  const link = element("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function renderCurve(run) {
  const m = run.metrics;
  setText(
    "probability-metrics",
    `Balanced accuracy ${percent(m.balanced_accuracy)} · Macro AUROC ${percent(m.macro_auroc)} · Log loss ${m.log_loss.toFixed(4)} · Multiclass Brier ${m.brier_score.toFixed(4)} · Tumour AUROC ${percent(m.tumour.auroc)} · Tumour AP ${percent(m.tumour.average_precision)}`,
  );
  const svg = $("coverage-curve"),
    ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren();
  const node = (tag, attrs, text) => {
    const n = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text) n.textContent = text;
    return n;
  };
  svg.append(
    node("path", { d: "M45 10V150H580", stroke: "currentColor", fill: "none" }),
  );
  for (const [x, y, text] of [
    [45, 170, "0%"],
    [550, 170, "100%"],
    [235, 187, "Retained coverage"],
    [5, 16, "100%"],
    [15, 150, "0%"],
  ])
    svg.append(node("text", { x, y }, text));
  const points = (m.review_curve || extendedMetrics(run.rows).review_curve)
    .filter((r) => r.retained_error !== null)
    .map((r) => `${45 + r.coverage * 535},${150 - r.retained_error * 140}`)
    .join(" ");
  svg.append(
    node("polyline", {
      points,
      stroke: "var(--accent)",
      "stroke-width": 2,
      fill: "none",
    }),
  );
  svg.append(node("text", { x: 55, y: 20 }, "Retained error rate"));
  const links = $("run-links");
  links.replaceChildren();
  for (const [file, label] of [
    ["predictions.csv", "Predictions CSV"],
    ["run_metadata.json", "Run metadata"],
  ]) {
    const a = element("a", "", label);
    a.href = `./data/${run.item.path}${file}`;
    links.append(a, document.createTextNode(" · "));
  }
}

function activateModel(id, fromURL = false) {
  const run = modelRuns.get(id);
  if (!run) return;
  state.rows = run.rows;
  state.metadata = run.metadata;
  state.summary = run.summary;
  $("model-select").value = id;
  setText(
    "model-note",
    run.item.historical
      ? "Historical fixed C=1 reference. Re-run ResNet-18 with validation-selected C for a controlled encoder comparison."
      : `${run.item.benchmark_version} · ${run.metadata.run_id} · Uncalibrated probabilities`,
  );
  const params = new URLSearchParams(location.search);
  if (fromURL) {
    state.review = readCutoff(
      params,
      "review",
      state.metadata.protocol.review_cutoff,
    );
    state.tumour = readCutoff(
      params,
      "tumour",
      state.metadata.protocol.tumour_threshold,
    );
  }
  renderSummary();
  renderThresholds();
  renderMatrix();
  renderRecall();
  renderConfidence();
  clearErrorFilters();
  renderCurve(run);
  setText(
    "load-status",
    `${run.item.label} · ${count(run.rows.length)} held-out patches · ${count(run.metadata.train_images)} development patches · Saved evaluation ${run.metadata.completed_at_utc.slice(0, 10)}`,
  );
}

async function start() {
  const [registryText, lookupText] = await Promise.all([
    fetchText("models.json"),
    fetchText("class_lookup.csv"),
  ]);
  registry = JSON.parse(registryText);
  if (
    registry.schema_version !== 1 ||
    new Set(registry.models.map((m) => m.id)).size !== registry.models.length
  )
    throw new Error("Invalid model registry.");
  const lookup = parseCSV(lookupText);
  if (
    lookup.length !== 9 ||
    new Set(lookup.map((r) => r.class_code)).size !== 9 ||
    lookup.some((r) => !CLASSES.includes(r.class_code))
  )
    throw new Error("The tissue lookup is incomplete.");
  state.lookup = new Map(lookup.map((r) => [r.class_code, r]));
  const available = registry.models.filter(
    (item) => item.status === "available",
  );
  const loaded = await Promise.allSettled(available.map(loadModel));
  loaded.forEach((result, i) => {
    if (result.status === "fulfilled")
      modelRuns.set(available[i].id, result.value);
    else {
      available[i].reason = `Validation failed: ${result.reason.message}`;
    }
  });
  const requested = new URLSearchParams(location.search).get("model");
  const selected =
    requested && modelRuns.has(requested) ? requested : registry.default;
  if (!modelRuns.has(selected))
    throw new Error(
      registry.models.find((m) => m.id === selected)?.reason ||
        "No valid completed model is available.",
    );
  for (const id of ["model-select", "compare-first", "compare-second"]) {
    $(id).replaceChildren();
    for (const item of registry.models) {
      const option = element(
        "option",
        "",
        item.label +
          (modelRuns.has(item.id) ? "" : ` — ${item.reason || "unavailable"}`),
      );
      option.value = item.id;
      option.disabled = !modelRuns.has(item.id);
      $(id).append(option);
    }
    $(id).value = selected;
    $(id).disabled = modelRuns.size < 2;
  }
  const comparable = [...modelRuns.values()].filter((r) => !r.item.historical);
  if (comparable.length >= 2) {
    $("compare-first").value = comparable[0].item.id;
    $("compare-second").value = comparable[1].item.id;
  }
  activateModel(selected, true);
  renderLeaderboard();
  renderComparison();
  bindControls();
  $("model-select").addEventListener("change", (event) => {
    activateModel(event.target.value);
    const url = new URL(location.href);
    url.searchParams.set("model", event.target.value);
    history.replaceState(null, "", url);
  });
  $("matrix-mode").addEventListener("change", () => {
    renderMatrix();
    renderSelection();
  });
  for (const id of ["compare-first", "compare-second"])
    $(id).addEventListener("change", renderComparison);
  $("comparison-category").addEventListener("change", renderComparisonCases);
  $("download-comparison").addEventListener("click", () =>
    downloadCSV(
      comparisonCases(),
      [
        "image_id",
        "true_class",
        "first_prediction",
        "second_prediction",
        "corrected",
        "regression",
        "both_wrong",
        "disagreement",
      ],
      "model-comparison.csv",
    ),
  );
  document.querySelectorAll("[data-sort]").forEach((button) =>
    button.addEventListener("click", () => {
      const key = button.dataset.sort;
      leaderboardSort = {
        key,
        descending:
          key === leaderboardSort.key
            ? !leaderboardSort.descending
            : key !== "label",
      };
      renderLeaderboard();
    }),
  );
  $("dashboard").hidden = false;
  $("load-status").classList.add("loaded");
}

start().catch((error) => {
  console.error(error);
  setText(
    "load-status",
    `The evaluation could not be loaded: ${error.message} Reload the page or check the data files in the repository.`,
  );
  $("load-status").classList.add("failed");
});
