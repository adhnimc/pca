# PCA and Multivariate Learning Tool

Static interactive app for comparing raw-value PCA with rank PCA, then extending that core into a broader multivariate learning tool.

## App location

The maintained repo app lives in [`app/v4`](/Users/adhni/Desktop/PCA/pca/app/v4).

The canonical standalone reference HTML is stored at [`app/v4/reference/full_reference.html`](/Users/adhni/Desktop/PCA/pca/app/v4/reference/full_reference.html).

## Syllabus coverage now included

### PCA core

- PCA on scaled raw values
- PCA on ranked values (Spearman-style PCA)
- Simulated dataset mode
- Real dataset switcher for Iris, Wine, and Breast Cancer Wisconsin
- Simulated-data controls for observations, noise, nonlinearity, outliers, outlier strength, and seed
- Raw vs rank PC1 and PC2 comparison plots
- Raw PCA biplot
- Rank / Spearman-style PCA biplot
- Pearson correlation heatmap
- Spearman correlation heatmap
- Loadings comparison table

### Distance and geometry

- Distance explorer with:
  - Euclidean distance
  - Manhattan distance
  - Pearson / correlation distance
  - Spearman-style / rank distance
- Distance matrix heatmaps on a readable sampled subset
- Plain-English summary of how raw-value geometry changes after ranking
- Classical MDS
- Raw-distance MDS plot
- Rank / correlation-distance MDS plot
- MDS vs PCA distance-fit comparison

### Clustering

- K-means clustering
- Hierarchical clustering
- Clustering on:
  - original scaled variables
  - first `k` raw PCA scores
  - first `k` rank PCA scores
- Cluster colouring on raw PCA and rank PCA plots
- Simple evaluation metrics:
  - silhouette score
  - within-cluster sum of squares
  - adjusted Rand index when labels are available

### Supervised models and validation

- Model comparison on:
  - original variables
  - first `k` raw PCs
  - first `k` rank PCs
- Included models:
  - KNN
  - logistic regression
  - discriminant analysis (LDA)
  - decision tree
- Reproducible train/test split
- Reproducible `k`-fold cross-validation
- Separate analysis seed for sampling, splitting, folds, and random initialisation

### Interpretation and related methods

- Basic XAI / interpretation via:
  - readable logistic coefficient importance
  - simple decision-tree importance and first-split rule
- Factor analysis section with a small PCA vs factor-analysis comparison
- Correspondence analysis demo with:
  - one built-in categorical contingency table
  - row / column map
  - explanation of when CA is used instead of PCA

## What is still left for later

- Model-based clustering
- SVM
- More advanced explainability methods beyond simple coefficient / tree summaries
- Broader correspondence-analysis dataset library
- Richer factor-analysis options such as rotation and tuning of factor count

## Main compromises

- The app stays fully static HTML/CSS/JS with no framework and no build step.
- Distance heatmaps, MDS, and clustering may use a reproducible subset of rows when the active dataset is large, so the browser remains responsive.
- Factor analysis is implemented as a lightweight principal-axis style approximation rather than a feature-complete FA package.
- Rank-PC resampling is intended for teaching and comparison, not as a production ML preprocessing pipeline.
- The focus is on compact, inspectable implementations rather than exhaustive modelling options.

## Structure

```text
app/
└── v4/
    ├── index.html
    ├── style.css
    ├── script.js
    ├── data/
    │   ├── iris.js
    │   ├── wine.js
    │   └── breast_cancer.js
    └── reference/
        └── full_reference.html
```

## Run locally

```bash
cd app/v4
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Notes

- The app is fully static. No framework or build step is required.
- `index.html`, `style.css`, and `script.js` now cover the original PCA comparison plus the additional syllabus topics listed above.
- The dataset payloads remain separated under [`app/v4/data`](/Users/adhni/Desktop/PCA/pca/app/v4/data) so future dataset changes do not require editing the main app logic.
