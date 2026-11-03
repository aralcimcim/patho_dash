import test from "node:test";
import assert from "node:assert/strict";
import { compareModels, validateExtended } from "../docs/benchmark.mjs";
import { fixtures } from "./benchmark-fixtures.mjs";

test("pair comparison aligns reordered IDs, separates corrected errors and regressions", () => {
  const { runs } = fixtures(),
    result = compareModels(...runs);
  assert.deepEqual(result.counts, {
    corrected: 2,
    regression: 1,
    both_wrong: 976,
    disagreement: 3,
  });
  assert.equal(result.cases.length, 7180);
  assert.equal(result.recall.length, 9);
  assert.throws(
    () => compareModels(runs[0], { ...runs[1], rows: runs[1].rows.slice(1) }),
    /image IDs/,
  );
  const altered = structuredClone(runs[1]);
  altered.rows[0].true_class = "ADI";
  assert.throws(() => compareModels(runs[0], altered), /true labels/);
  const version = structuredClone(runs[1]);
  version.metadata.benchmark_version = "different";
  assert.throws(() => compareModels(runs[0], version), /benchmark version/);
  const duplicate = structuredClone(runs[0]);
  duplicate.rows[1] = duplicate.rows[0];
  assert.throws(() => compareModels(duplicate, runs[1]), /unique image IDs/);
});

test("extended metric checks reject altered aggregate scores", () => {
  const { runs, files } = fixtures();
  const saved = JSON.parse(files.get("runs/test-encoder-0/metrics.json"));
  validateExtended(runs[0].rows, saved);
  saved.per_class[0].auroc = 0.1;
  assert.throws(() => validateExtended(runs[0].rows, saved), /Saved metric/);
});
