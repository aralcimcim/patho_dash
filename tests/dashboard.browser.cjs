const { test, expect } = require("@playwright/test");
const fs = require("node:fs/promises");

async function open(page, suffix = "") {
  await page.goto(`./${suffix}`);
  await expect(page.locator("#dashboard")).toBeVisible();
}

test("saved benchmarks, complete matrix and charts load without browser errors", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await open(page);
  for (const [id, value] of Object.entries({
    "evaluated-images": "7,180",
    "overall-accuracy": "86.38%",
    "macro-f1": "82.77%",
    "high-confidence-errors": "460",
    "high-confidence-rate": "7.52%",
    "review-flagged": "713",
    "review-proportion": "9.93%",
    "review-coverage": "90.07%",
    "retained-accuracy": "90.69%",
    "tumour-sensitivity": "82.40%",
    "tumour-specificity": "98.35%",
    "tumour-precision": "91.20%",
  }))
    await expect(page.locator(`#${id}`)).toHaveText(value);
  await expect(page.locator(".matrix-count")).toHaveCount(81);
  await expect(page.locator(".recall-row")).toHaveCount(9);
  await expect(page.locator(".confidence-bin")).toHaveCount(10);
  await expect(page.locator("#error-scope")).toContainText("978 of 978 errors");
  await expect(page.locator("#error-rows tr")).toHaveCount(20);
  await page.locator(".provenance summary").click();
  await expect(page.locator("#benchmark-status")).toContainText("54 of 54");
  await expect(page.locator("#benchmark-rows tr")).toHaveCount(54);
  await expect(page.locator("#benchmark-rows tr td:last-child")).toHaveText(
    Array(54).fill("Pass"),
  );
  expect(errors).toEqual([]);
});

test("independent cutoffs reproduce Power BI observations, endpoints and bookmarks", async ({
  page,
}) => {
  await open(page);
  await page.locator("#review-value").fill("0.35");
  await expect(page.locator("#review-flagged")).toHaveText("3");
  await expect(page.locator("#review-proportion")).toHaveText("0.04%");
  await expect(page.locator("#review-coverage")).toHaveText("99.96%");
  await expect(page.locator("#retained-accuracy")).toHaveText("86.40%");
  await expect(page.locator("#tumour-sensitivity")).toHaveText("82.40%");
  await page.locator("#tumour-value").fill("0.66");
  await expect(page.locator("#tumour-sensitivity")).toHaveText("79.97%");
  await expect(page.locator("#tumour-specificity")).toHaveText("98.52%");
  await expect(page.locator("#review-flagged")).toHaveText("3");
  await expect(page.locator("#overall-accuracy")).toHaveText("86.38%");
  await expect(page.locator("#high-confidence-errors")).toHaveText("460");
  await expect(page).toHaveURL(/review=0.35&tumour=0.66/);
  await page.reload();
  await expect(page.locator("#review-value")).toHaveValue("0.35");
  await expect(page.locator("#tumour-value")).toHaveValue("0.66");
  await page.locator("#review-value").fill("0.54");
  await expect(page.locator("#review-flagged")).toHaveText("148");
  await page.locator("#review-value").fill("0");
  await expect(page.locator("#review-coverage")).toHaveText("100.00%");
  await page.locator("#tumour-value").fill("0");
  await expect(page.locator("#tumour-sensitivity")).toHaveText("100.00%");
  await expect(page.locator("#tumour-specificity")).toHaveText("0.00%");
  await page.locator("#reset-defaults").click();
  await expect(page.locator("#review-value")).toHaveValue("0.80");
  await expect(page.locator("#tumour-value")).toHaveValue("0.50");
  await open(page, "?review=&tumour=invalid");
  await expect(page.locator("#review-value")).toHaveValue("0.80");
  await expect(page.locator("#tumour-value")).toHaveValue("0.50");
});

test("matrix, recall, confidence and search inspect only the error table; downloads contain the selected rows", async ({
  page,
}) => {
  await open(page);
  await page
    .locator('.matrix-count[data-true-class="STR"][data-predicted-class="MUS"]')
    .click();
  await expect(page.locator("#error-scope")).toContainText(
    "143 of 978 errors · true STR · predicted MUS",
  );
  const downloadEvent = page.waitForEvent("download");
  await page.locator("#download-errors").click();
  const download = await downloadEvent;
  const csv = await fs.readFile(await download.path(), "utf8");
  const lines = csv.trim().split(/\r?\n/);
  expect(lines).toHaveLength(144);
  expect(lines[0]).toContain(
    "image_id,dataset_split,model_name,true_class,predicted_class,confidence",
  );
  const { parseCSV } = await import("../docs/metrics.mjs");
  const downloadedRows = parseCSV(csv);
  expect(
    downloadedRows.every(
      (row) => row.true_class === "STR" && row.predicted_class === "MUS",
    ),
  ).toBe(true);
  await page.locator("#next-page").click();
  await expect(page.locator("#error-page-status")).toHaveText("Page 2 of 8");
  await page.locator('.recall-row[data-class-code="STR"]').click();
  await expect(page.locator("#error-scope")).toContainText(
    "226 of 978 errors · true STR",
  );
  await page.locator("#clear-errors").click();
  await page.locator('.confidence-bin[data-bin="9"]').click();
  await expect(page.locator("#error-scope")).toContainText("460 of 978 errors");
  await page.locator("#clear-errors").click();
  await page.locator("#high-errors-only").check();
  await expect(page.locator("#error-scope")).toContainText("460 of 978 errors");
  await page.locator("#error-search").fill("no-such-patch");
  await expect(page.locator("#download-errors")).toBeDisabled();
  await expect(page.locator("#error-rows")).toContainText(
    "No misclassified patches match",
  );
  await page.locator("#reset-defaults").click();
  await expect(page.locator("#error-scope")).toContainText("978 of 978 errors");
  await page.locator("#sort-confidence").click();
  await expect(page.locator("#confidence-sort-header")).toHaveAttribute(
    "aria-sort",
    "ascending",
  );
  await expect(page.locator("#overall-accuracy")).toHaveText("86.38%");
  await expect(page.locator("#evaluated-images")).toHaveText("7,180");
  await expect(page.locator("#review-flagged")).toHaveText("713");
});

test("keyboard navigation works and invalid inputs keep the last valid threshold", async ({
  page,
}) => {
  await open(page);
  await page.keyboard.press("Tab");
  await expect(page.locator(".skip-link")).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main")).toBeFocused();
  await page.locator("#review-range").focus();
  await page.keyboard.press("Home");
  await expect(page.locator("#review-flagged")).toHaveText("0");
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#review-value")).toHaveValue("0.01");
  await page.locator("#review-value").fill("1.5");
  await page.locator("h1").click();
  await expect(page.locator("#review-value")).toHaveValue("0.01");
});

for (const width of [375, 768, 1440]) {
  test(`dashboard fits ${width}px with its tables scrolling inside panels`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 1100 });
    await open(page);
    await page.emulateMedia({ reducedMotion: "reduce" });
    const sizes = await page.evaluate(() => ({
      content: document.documentElement.scrollWidth,
      viewport: innerWidth,
    }));
    expect(sizes.content).toBeLessThanOrEqual(sizes.viewport);
    if (width === 375) {
      const scroll = page.getByRole("region", {
        name: "Confusion matrix, scroll horizontally on small screens",
      });
      expect(
        await scroll.evaluate((el) => el.scrollWidth > el.clientWidth),
      ).toBe(true);
      await page.locator("#review-value").fill("0.35");
      await expect(page.locator("#review-flagged")).toHaveText("3");
      await page.locator("#reset-defaults").click();
    }
    await page.screenshot({
      path: `/tmp/pathology-dashboard-${width}.png`,
      fullPage: true,
    });
    if (width === 1440)
      await page.screenshot({ path: "/tmp/pathology-dashboard-preview.png" });
  });
}

test("missing files and altered predictions show a clear error instead of unverified results", async ({
  page,
}) => {
  await page.route("**/data/model_metrics.csv", (route) =>
    route.fulfill({ status: 404, body: "Missing" }),
  );
  await page.goto("./");
  await expect(page.locator("#load-status")).toContainText(
    "could not be loaded",
  );
  await expect(page.locator("#dashboard")).toBeHidden();
  await page.unroute("**/data/model_metrics.csv");
  await page.route("**/data/predictions.csv", async (route) => {
    const response = await route.fetch();
    await route.fulfill({
      response,
      body: (await response.text()).replace("image_id", "changed_id"),
    });
  });
  await page.goto("./");
  await expect(page.locator("#load-status")).toContainText("checksum differs");
  await expect(page.locator("#dashboard")).toBeHidden();
});
