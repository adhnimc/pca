# PCA: Raw Values vs Rank PCA

Static interactive app for comparing:

- PCA on scaled raw values
- PCA on ranked values (Spearman-style PCA)

## App location

The maintained repo app lives in [`app/v4`](/Users/adhni/Desktop/PCA/pca/app/v4).

The canonical standalone reference HTML is stored at [`app/v4/reference/full_reference.html`](/Users/adhni/Desktop/PCA/pca/app/v4/reference/full_reference.html).

## Included features

- Simulated dataset mode
- Real dataset switcher for Iris, Wine, and Breast Cancer Wisconsin
- Simulated-data controls for observations, noise, nonlinearity, outliers, outlier strength, and seed
- Preset buttons
- Toggle for coloring outliers or classes
- Toggle for loading labels
- PC1 comparison plot
- PC2 comparison plot
- Raw PCA biplot
- Rank / Spearman-style PCA biplot
- Pearson correlation heatmap
- Spearman correlation heatmap
- Loadings comparison table

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
- `index.html`, `style.css`, and `script.js` preserve the reference UI and behavior while splitting the app into maintainable repo files.
- The dataset payloads are separated under [`app/v4/data`](/Users/adhni/Desktop/PCA/pca/app/v4/data) so future dataset changes do not require editing the main app logic.
