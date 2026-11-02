const path = require("node:path");
const { defineConfig } = require("@playwright/test");

module.exports = defineConfig({
  testDir: __dirname,
  testMatch: "dashboard.browser.cjs",
  workers: 1,
  timeout: 30000,
  reporter: "list",
  use: {
    baseURL: process.env.PATHOLOGY_BASE_URL || "http://127.0.0.1:4179/",
    browserName: "chromium",
    viewport: { width: 1440, height: 1100 },
    launchOptions: {
      executablePath: process.env.PATHOLOGY_BROWSER_PATH || undefined,
    },
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: process.env.PATHOLOGY_BASE_URL
    ? undefined
    : {
        command:
          "python3 -m http.server 4179 --bind 127.0.0.1 --directory docs",
        cwd: path.resolve(__dirname, ".."),
        url: "http://127.0.0.1:4179/",
        reuseExistingServer: false,
        timeout: 10000,
      },
});
