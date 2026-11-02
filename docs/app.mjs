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
      const button = element("button", "matrix-count", count(n));
      button.type = "button";
      button.dataset.trueClass = CLASSES[i];
      button.dataset.predictedClass = CLASSES[j];
      button.setAttribute(
        "aria-label",
        `True ${CLASSES[i]}, predicted ${CLASSES[j]}: ${count(n)} patches${i === j ? ", correct predictions" : ", inspect errors"}`,
      );
      button.title = `${CLASSES[i]} → ${CLASSES[j]}: ${count(n)} patches`;
      const strength = maximum ? n / maximum : 0;
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

async function start() {
  const [predictionsText, metricsText, lookupText, metadataText] =
    await Promise.all([
      fetchText("predictions.csv"),
      fetchText("model_metrics.csv"),
      fetchText("class_lookup.csv"),
      fetchText("run_metadata.json"),
    ]);
  const metadata = JSON.parse(metadataText);
  if (globalThis.crypto?.subtle) {
    const bytes = await crypto.subtle.digest(
      "SHA-256",
      new TextEncoder().encode(predictionsText),
    );
    const actual = [...new Uint8Array(bytes)]
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
    if (actual !== metadata.predictions_sha256)
      throw new Error(
        "Prediction checksum differs from the saved run metadata.",
      );
  }
  state.metadata = metadata;
  state.rows = preparePredictions(parseCSV(predictionsText), metadata);
  const lookup = parseCSV(lookupText);
  if (
    lookup.length !== 9 ||
    new Set(lookup.map((row) => row.class_code)).size !== 9 ||
    lookup.some((row) => !CLASSES.includes(row.class_code))
  )
    throw new Error("The tissue lookup is incomplete.");
  state.lookup = new Map(lookup.map((row) => [row.class_code, row]));
  state.summary = summarize(state.rows);
  const checks = checkBenchmarks(state.rows, parseCSV(metricsText), metadata);
  if (checks.some((check) => !check.matches))
    throw new Error(
      "Browser calculations disagree with the fixed Python benchmarks. Please check the source files.",
    );
  const params = new URLSearchParams(location.search);
  state.review = readCutoff(params, "review", metadata.protocol.review_cutoff);
  state.tumour = readCutoff(
    params,
    "tumour",
    metadata.protocol.tumour_threshold,
  );
  renderSummary();
  renderThresholds();
  renderMatrix();
  renderRecall();
  renderConfidence();
  renderErrors();
  bindControls();
  $("dashboard").hidden = false;
  setText(
    "load-status",
    `${count(state.rows.length)} held-out patches · ${count(metadata.train_images)} training patches · Saved evaluation from ${new Date(metadata.completed_at_utc).toISOString().slice(0, 10)}`,
  );
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
