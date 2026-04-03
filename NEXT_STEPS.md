# Next Steps for `app/v4`

This file is a handoff note for future sessions. It is written to be actionable without needing to reconstruct the recent work from scratch.

## Current State

The maintained app is [`app/v4`](/Users/adhni/Desktop/PCA/pca/app/v4).

It is a static HTML/CSS/JS app. Do not replace it with a framework and do not rebuild from scratch.

Recent completed work:

- Preserved the original PCA comparison app.
- Expanded it into a broader multivariate learning tool covering:
  - raw PCA vs rank PCA
  - simulated + real datasets
  - PC1 / PC2 comparison plots
  - raw and rank biplots
  - Pearson and Spearman heatmaps
  - loadings table
  - distance explorer
  - classical MDS
  - K-means + hierarchical clustering
  - supervised models:
    - KNN
    - logistic regression
    - LDA
    - decision tree
  - train/test split + k-fold CV
  - factor analysis comparison
  - correspondence analysis demo
  - simple XAI / interpretation
- Added UX improvements so the app is easier to explore:
  - clearer navigation
  - grouped controls
  - guidance text
  - contextual disabling of irrelevant controls
  - better narrow-layout table handling

Important recent commits:

- `711b7f7` Expand v4 into a broader multivariate learning tool
- `a3a4e6b` Polish v4 exploration UX

## Non-Negotiable Constraints

These should be treated as hard constraints unless the user explicitly changes them.

- Work in the existing repo.
- The maintained app is `app/v4`.
- Keep it static HTML/CSS/JS.
- Do not switch frameworks.
- Preserve the current PCA app and build around it.
- Do not overbuild.
- Prefer minimal finished features over ambitious unfinished ones.
- Keep the UI coherent with the current v4 style.
- Reuse existing helpers where practical.
- Refactor only when it makes extension clearly easier.

## What Has Not Been Fully Closed Yet

These are the main open areas.

### 1. Real browser QA is still needed

Code-path checks and fake-DOM execution checks have been run, but a true browser-level pass is still the next highest-value task.

What still needs direct visual/browser confirmation:

- desktop layout
- narrow-width/mobile layout
- plot label crowding
- navigation flow
- scroll behavior
- section readability
- perceived performance while dragging controls
- whether captions feel too dense or too vague

### 2. Deferred syllabus items

These are still left for later:

- SVM
- model-based clustering

Model-based clustering should only be added if it can be done cleanly in static JS without turning the app brittle or opaque.

### 3. Final teaching polish

The app is already usable, but it can still improve in how it teaches users what to notice.

Good future improvements:

- sharper “what to look for” prompts
- better defaults per dataset
- slightly tighter wording in dense captions
- clearer distinction between:
  - class labels
  - outlier colouring
  - cluster assignments

## Recommended Next Priority Order

Follow this order unless the user redirects.

1. Run a real browser QA pass on `app/v4`
2. Fix any UX or readability issues found in the browser
3. Add `SVM` to the supervised-model section
4. Reassess whether model-based clustering is worth implementing
5. Do a final documentation consistency pass

## Browser QA Checklist

Run the app locally:

```bash
cd /Users/adhni/Desktop/PCA/pca/app/v4
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000
```

### Datasets to test

- `simulated`
- `iris`
- `wine`
- `breast_cancer`

### Controls to test

- dataset switcher
- simulated-only controls
- analysis seed
- distance metric
- distance heatmap sample size
- MDS / clustering sample size
- PC count
- clustering method
- clustering feature space
- cluster count
- validation mode
- holdout test fraction
- CV folds
- color toggle
- loading-label toggle
- preset buttons

### Browser QA questions

- Does the page read in a sensible order without explanation from us?
- Are disabled controls obvious and intuitive?
- Is it clear which settings affect which sections?
- Does any section feel too slow or too busy?
- Are the plot captions helpful or noisy?
- Do any tables overflow awkwardly?
- Do any SVG labels become unreadable?
- Does the guidance text improve exploration or just add clutter?

### Specific scenario checks

#### Simulated

- Try the preset buttons.
- Confirm it is clear that:
  - point colouring shows outliers
  - supervised labels come from latent low/mid/high groups
- Check whether the nonlinear and outlier-heavy presets produce clearly different raw-vs-rank behavior.

#### Iris

- Confirm this still feels like the cleanest “first real example”.
- Check whether clustering and supervised sections are easy to interpret.

#### Wine

- Check readability under more variables and more overlap.
- Watch for plot clutter and overly dense interpretation text.

#### Breast cancer

- Confirm the supervised section is readable and not overwhelming.
- Check whether the two-class case makes cluster summaries and metrics easy to understand.

### Mobile / narrow layout checks

- width around `900px`
- width around `640px`
- width narrower than `480px`

Things to watch:

- stacked section readability
- control card height / scrolling
- table horizontal scrolling
- whether charts remain legible

## Expected Immediate Fixes After Browser QA

Do not assume all of these are needed. Only apply them if the browser pass justifies them.

- reduce label density on the busiest plots
- shorten a few captions
- improve section spacing on smaller screens
- refine guide text if it feels too wordy
- change defaults where a better teaching path becomes obvious
- simplify any control labels that feel technical for no benefit

## SVM Implementation Plan

`SVM` is the best next syllabus addition after browser QA.

### Goal

Add a minimal, coherent `SVM` row to the supervised-model comparison.

### Requirements

- Keep it in the existing supervised-model section.
- Compare on:
  - original variables
  - first `k` raw PCs
  - first `k` rank PCs
- Reuse the existing validation pipeline.
- Reuse the existing analysis seed behavior.
- Add a short plain-English explanation of what SVM is doing.

### Preferred implementation shape

- Keep it compact and transparent.
- If a linear SVM is much easier to implement clearly than a more flexible kernel version, prefer a linear SVM first.
- Avoid adding a large opaque solver unless there is a very strong reason.

### Do not do this

- Do not build a giant ML subsystem.
- Do not turn the supervised section into a configuration overload.
- Do not add five SVM variants.

## Model-Based Clustering Decision Gate

Treat this as optional.

Add it only if these are true:

- the implementation is understandable
- runtime remains acceptable in-browser
- the UI stays simple
- it adds real syllabus value beyond current clustering

If those conditions are not met, leave it out and document that decision clearly.

## Files Most Likely To Change Next

- [`app/v4/index.html`](/Users/adhni/Desktop/PCA/pca/app/v4/index.html)
- [`app/v4/style.css`](/Users/adhni/Desktop/PCA/pca/app/v4/style.css)
- [`app/v4/script.js`](/Users/adhni/Desktop/PCA/pca/app/v4/script.js)
- [`README.md`](/Users/adhni/Desktop/PCA/pca/README.md)

## Verification Commands Used So Far

These have already been useful and should remain the default quick checks.

Syntax check:

```bash
node --check app/v4/script.js
```

Serve locally:

```bash
cd app/v4
python3 -m http.server 8000
```

## Definition of Done For The Next Session

The next session should aim to leave the app in this state:

- browser QA completed on real rendered pages
- any UX issues found are fixed
- the app still feels lightweight and easy to explore
- `SVM` is added cleanly, or there is a clear written reason it was deferred
- README remains accurate

## Practical Session Start Prompt

If a future session needs a concise starting objective, use this:

> Work in the existing repo. The maintained app is `app/v4`. Keep static HTML/CSS/JS. Start with a real browser QA pass of the current app, fix any user-facing exploration issues you find, then add a minimal `SVM` model to the supervised section using the existing validation pipeline and seed behavior. Do not overbuild.
