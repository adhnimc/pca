# PCA: Raw Values vs Rank PCA

Interactive app for comparing:

- PCA on scaled raw values
- PCA on ranked values (Spearman-style PCA)

## Included features

- Simulated dataset with controls for noise, nonlinearity, and outliers
- Real built-in datasets:
  - Iris
  - Wine
  - Breast Cancer Wisconsin
- PC1 comparison
- PC2 comparison
- Raw PCA biplot
- Rank PCA biplot
- Pearson correlation heatmap
- Spearman correlation heatmap
- Loadings table

## Structure

```text
pca-raw-vs-rank-repo/
├── app/
│   └── index.html
├── README.md
└── .gitignore
```

## Run locally

Because this is a self-contained HTML app, you can simply open:

`app/index.html`

Or serve it locally:

### Python
```bash
cd app
python3 -m http.server 8000
```

Then open:
`http://localhost:8000`

## Git setup

Initialize git:

```bash
git init
git add .
git commit -m "Initial commit: PCA raw vs rank interactive app"
```

Connect to GitHub:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/pca-raw-vs-rank.git
git push -u origin main
```

## Suggested next upgrades

- Upload your own CSV
- Toggle between Pearson / Spearman / Kendall comparison
- Scree plot
- Explained variance chart
- Brushing to inspect selected points
- Download current simulated dataset as CSV
- More real datasets
- Robust PCA comparison
- Polychoric / ordinal alternatives for true ordered categories

## Versioning habit

A good workflow is:

1. make one small change
2. test it
3. commit with a clear message

Example:

```bash
git add .
git commit -m "Add CSV upload support"
```
