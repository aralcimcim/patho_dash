# Deploy and update the dashboard

This repository serves a static website from `docs/`. Everything runs in the visitor's browser using the committed CSVs. No API key, Python application host, Power BI licence or model-serving backend is needed.

## GitHub Pages

1. Push the repository to GitHub.
2. Open **Settings → Pages**.
3. Set **Source → Deploy from a branch**.
4. Select **main**, folder **/docs**, then save.
5. Wait for the **pages build and deployment** action to complete.

The project URL is <https://aralcimcim.github.io/german-pathology-ai/>. Keeping `.nojekyll` in `docs/` prevents Jekyll processing. Asset and data paths are relative, including when the site is hosted below a repository name.

The published data are derived from the cited public CC BY 4.0 dataset. Attribution and processing descriptions are included in the app and repository. Original histology images are not uploaded.

## Update a result snapshot

```bash
python3 scripts/import_results.py /path/to/extracted/results
npm test
npm run test:browser
```

Update README metrics and the real browser screenshot if the run changes. Tests intentionally pin the current snapshot's counts and metrics: inspect any differences and update those expectations only after validating a new run. The dashboard's own benchmark comparison reads the new saved metrics automatically.

Review `git diff --stat`, stage the intended data/source changes, commit and push. GitHub Pages republishes from `main`. Check the deployment action and reload the live page after it finishes.

## Local preview

```bash
npm run serve
```

Open <http://127.0.0.1:4179/>. Stop the server with Ctrl+C. Directly opening the HTML file cannot fetch the CSVs.

## Checks

`npm test` checks the real prediction checksum, all fixed exported benchmarks, all confusion cells, cutoff endpoints, denominator semantics, confidence bins, CSV handling and invalid input.

For browser tests, run `npm ci`, `npx playwright install chromium`, then `npm run test:browser`. An existing Chrome installation can be used with `PATHOLOGY_BROWSER_PATH=/path/to/google-chrome`. The test server stops when Playwright finishes.

To check a deployed instance with the same browser suite, set `PATHOLOGY_BASE_URL=https://aralcimcim.github.io/german-pathology-ai/`. Tests use fresh browser contexts; the app itself does not persist a user's inspection selections.

The CI workflow runs the numerical benchmark tests for every push and pull request. Browser tests are run separately before publishing interface changes.
