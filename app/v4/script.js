(function() {
  const REAL_DATASETS = window.PCA_DATASETS || {};
  const RANGE_IDS = [
    "n", "noise", "nonlinear", "outliers", "outlierStrength", "seed",
    "analysisSeed", "heatmapSize", "analysisSubset", "pcCount",
    "clusterK", "testFraction", "cvFolds"
  ];
  const SELECT_IDS = [
    "datasetMode", "distanceMetric", "clusterMethod", "clusterSpace", "validationMode"
  ];
  const TOGGLE_IDS = ["colorSpecial", "showLabels"];
  const palette = [
    "rgba(31,119,180,0.84)",
    "rgba(220,38,38,0.84)",
    "rgba(5,150,105,0.84)",
    "rgba(217,119,6,0.84)",
    "rgba(124,58,237,0.84)",
    "rgba(8,145,178,0.84)"
  ];
  const CA_DEMO = {
    name: "Study major vs favourite assessment",
    rowLabels: ["Arts", "Business", "Science", "Health"],
    colLabels: ["Essay", "Presentation", "Project", "Exam"],
    counts: [
      [48, 36, 22, 18],
      [26, 24, 41, 39],
      [12, 18, 44, 52],
      [18, 20, 28, 46]
    ]
  };
  let updateTimer = null;

  RANGE_IDS.forEach(bindRangePair);
  SELECT_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", scheduleUpdate);
  });
  TOGGLE_IDS.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", scheduleUpdate);
  });
  document.getElementById("rerun").addEventListener("click", update);

  document.getElementById("presetLinear").addEventListener("click", () => {
    setVals({noise: 0.25, nonlinear: 0.10, outliers: 2, outlierStrength: 2.0});
  });
  document.getElementById("presetNonlinear").addEventListener("click", () => {
    setVals({noise: 0.30, nonlinear: 1.00, outliers: 4, outlierStrength: 3.0});
  });
  document.getElementById("presetOutliers").addEventListener("click", () => {
    setVals({noise: 0.35, nonlinear: 0.70, outliers: 20, outlierStrength: 9.0});
  });

  function bindRangePair(id) {
    const slider = document.getElementById(id);
    const num = document.getElementById(id + "_num");
    if (!slider || !num) return;
    slider.addEventListener("input", () => {
      num.value = slider.value;
      scheduleUpdate();
    });
    num.addEventListener("input", () => {
      slider.value = num.value;
      scheduleUpdate();
    });
  }

  function scheduleUpdate() {
    clearTimeout(updateTimer);
    updateTimer = setTimeout(update, 60);
  }

  function setVals(obj) {
    Object.entries(obj).forEach(([k, v]) => {
      const slider = document.getElementById(k);
      const num = document.getElementById(k + "_num");
      if (slider) slider.value = v;
      if (num) num.value = v;
    });
    scheduleUpdate();
  }

  function getVal(id) {
    return parseFloat(document.getElementById(id).value);
  }

  function clamp(x, lo, hi) {
    return Math.min(hi, Math.max(lo, x));
  }

  function sum(arr) {
    return arr.reduce((a, b) => a + b, 0);
  }

  function mean(arr) {
    return sum(arr) / arr.length;
  }

  function std(arr) {
    const m = mean(arr);
    const v = arr.reduce((s, x) => s + (x - m) * (x - m), 0) / Math.max(1, arr.length - 1);
    return Math.sqrt(v);
  }

  function trace(A) {
    return A.reduce((s, row, i) => s + row[i], 0);
  }

  function median(arr) {
    const sorted = arr.slice().sort((a, b) => a - b);
    const n = sorted.length;
    if (!n) return 0;
    if (n % 2 === 1) return sorted[(n - 1) / 2];
    return 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  }

  function quantile(arr, q) {
    const sorted = arr.slice().sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos);
    const hi = Math.ceil(pos);
    if (lo === hi) return sorted[lo];
    const w = pos - lo;
    return sorted[lo] * (1 - w) + sorted[hi] * w;
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function norm(v) {
    return Math.sqrt(dot(v, v));
  }

  function transpose(X) {
    return X[0].map((_, j) => X.map(row => row[j]));
  }

  function subsetRows(X, indices) {
    return indices.map(i => X[i].slice());
  }

  function subsetVector(v, indices) {
    return indices.map(i => v[i]);
  }

  function cloneMatrix(A) {
    return A.map(row => row.slice());
  }

  function identityMatrix(n) {
    return Array.from({length: n}, (_, i) =>
      Array.from({length: n}, (_, j) => (i === j ? 1 : 0))
    );
  }

  function safeCorr(a, b) {
    const ma = mean(a);
    const mb = mean(b);
    let num = 0;
    let va = 0;
    let vb = 0;
    for (let i = 0; i < a.length; i++) {
      const da = a[i] - ma;
      const db = b[i] - mb;
      num += da * db;
      va += da * da;
      vb += db * db;
    }
    const denom = Math.sqrt(va * vb);
    return denom > 1e-12 ? num / denom : 0;
  }

  function zscore(arr) {
    const m = mean(arr);
    const s = std(arr) || 1;
    return arr.map(x => (x - m) / s);
  }

  function fitScaler(X) {
    const cols = transpose(X);
    return {
      means: cols.map(mean),
      stds: cols.map(col => std(col) || 1)
    };
  }

  function transformScaler(X, scaler) {
    return X.map(row => row.map((x, j) => (x - scaler.means[j]) / scaler.stds[j]));
  }

  function standardizeColumns(X) {
    return transformScaler(X, fitScaler(X));
  }

  function covarianceMatrix(X) {
    const n = X.length;
    const p = X[0].length;
    const cov = Array.from({length: p}, () => Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = i; j < p; j++) {
        let s = 0;
        for (let r = 0; r < n; r++) s += X[r][i] * X[r][j];
        const val = s / Math.max(1, n - 1);
        cov[i][j] = val;
        cov[j][i] = val;
      }
    }
    return cov;
  }

  function lowerBound(sorted, x) {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (sorted[mid] < x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function upperBound(sorted, x) {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (sorted[mid] <= x) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  function rankArray(arr) {
    const indexed = arr.map((v, i) => ({v, i})).sort((a, b) => a.v - b.v);
    const ranks = new Array(arr.length);
    let i = 0;
    while (i < indexed.length) {
      let j = i;
      while (j + 1 < indexed.length && indexed[j + 1].v === indexed[i].v) j++;
      const avgRank = (i + j + 2) / 2;
      for (let k = i; k <= j; k++) ranks[indexed[k].i] = avgRank;
      i = j + 1;
    }
    return ranks;
  }

  function rankColumns(X) {
    return transpose(transpose(X).map(rankArray));
  }

  function rankColumnsNormalized(X) {
    return transpose(transpose(X).map(col => {
      const ranks = rankArray(col);
      return ranks.map(r => r / (col.length + 1));
    }));
  }

  function fitRankReference(X) {
    return {
      sortedColumns: transpose(X).map(col => col.slice().sort((a, b) => a - b)),
      n: X.length
    };
  }

  function transformRankReference(X, ref) {
    return transpose(transpose(X).map((col, j) => {
      const sorted = ref.sortedColumns[j];
      return col.map(x => {
        const left = lowerBound(sorted, x);
        const right = upperBound(sorted, x);
        if (!sorted.length) return 0.5;
        return clamp((left + right) / (2 * sorted.length), 0, 1);
      });
    }));
  }

  function mulberry32(a) {
    return function() {
      let t = a += 0x6D2B79F5;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randn(rng) {
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function shuffleInPlace(arr, rng) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }

  function orthogonalize(v, basis) {
    const out = v.slice();
    basis.forEach(b => {
      const proj = dot(out, b);
      for (let i = 0; i < out.length; i++) out[i] -= proj * b[i];
    });
    return out;
  }

  function powerIterationSymmetric(A, basis, seedOffset) {
    const rng = mulberry32(1000 + seedOffset);
    let v = Array.from({length: A.length}, () => rng() * 2 - 1);
    v = orthogonalize(v, basis);
    let nrm = norm(v);
    if (nrm < 1e-12) {
      v = Array(A.length).fill(0);
      v[seedOffset % A.length] = 1;
      nrm = 1;
    }
    v = v.map(x => x / nrm);

    for (let iter = 0; iter < 800; iter++) {
      const Av = A.map(row => dot(row, v));
      const ortho = orthogonalize(Av, basis);
      const len = norm(ortho);
      if (len < 1e-12) break;
      const next = ortho.map(x => x / len);
      let diff = 0;
      for (let i = 0; i < next.length; i++) diff += Math.abs(next[i] - v[i]);
      v = next;
      if (diff < 1e-11) break;
    }

    const lambda = dot(v, A.map(row => dot(row, v)));
    return {vector: v, value: lambda};
  }

  function topEigenPairs(A, k) {
    const basis = [];
    const values = [];
    for (let i = 0; i < Math.min(k, A.length); i++) {
      const eig = powerIterationSymmetric(A, basis, i + 1);
      basis.push(eig.vector.slice());
      values.push(eig.value);
    }
    return {vectors: basis, values};
  }

  function projectRows(X, components, k) {
    return X.map(row => components.slice(0, k).map(vec => dot(row, vec)));
  }

  function pcaFromStandardized(Xs, k) {
    const cov = covarianceMatrix(Xs);
    const eig = topEigenPairs(cov, Math.min(k, cov.length));
    const totalVar = trace(cov) || 1;
    const variance = eig.values.map(v => v / totalVar);
    const scores = eig.vectors.map(vec => Xs.map(row => dot(row, vec)));
    return {
      Xs,
      cov,
      components: eig.vectors.map(v => v.slice()),
      eigenvalues: eig.values.slice(),
      scores: scores.map(v => v.slice()),
      variance
    };
  }

  function pca(X, k) {
    const Xs = standardizeColumns(X);
    return pcaFromStandardized(Xs, k);
  }

  function alignPca(reference, candidate) {
    const out = {
      Xs: candidate.Xs.map(row => row.slice()),
      cov: candidate.cov.map(row => row.slice()),
      components: candidate.components.map(row => row.slice()),
      eigenvalues: candidate.eigenvalues.slice(),
      scores: candidate.scores.map(row => row.slice()),
      variance: candidate.variance.slice()
    };
    const limit = Math.min(reference.scores.length, out.scores.length);
    for (let k = 0; k < limit; k++) {
      if (safeCorr(reference.scores[k], out.scores[k]) < 0) {
        out.scores[k] = out.scores[k].map(v => -v);
        out.components[k] = out.components[k].map(v => -v);
      }
    }
    return out;
  }

  function matrixInverse(A) {
    const n = A.length;
    const M = A.map((row, i) => row.concat(identityMatrix(n)[i]));
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) {
        if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
      }
      if (Math.abs(M[pivot][col]) < 1e-12) return null;
      if (pivot !== col) {
        const tmp = M[col];
        M[col] = M[pivot];
        M[pivot] = tmp;
      }
      const div = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= div;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = M[r][col];
        if (factor === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= factor * M[col][j];
      }
    }
    return M.map(row => row.slice(n));
  }

  function safeInverseWithRidge(A) {
    for (let ridge = 1e-8; ridge <= 1; ridge *= 10) {
      const shifted = A.map((row, i) => row.map((x, j) => x + (i === j ? ridge : 0)));
      const inv = matrixInverse(shifted);
      if (inv) return inv;
    }
    return matrixInverse(identityMatrix(A.length));
  }

  function pcaLoadings(pcaObj, k) {
    const p = pcaObj.components[0].length;
    const loadings = Array.from({length: p}, () => Array(k).fill(0));
    for (let c = 0; c < k; c++) {
      const scale = Math.sqrt(Math.max(pcaObj.eigenvalues[c], 0));
      for (let i = 0; i < p; i++) {
        loadings[i][c] = pcaObj.components[c][i] * scale;
      }
    }
    return loadings;
  }

  function factorAnalysisApprox(X, k) {
    const Xs = standardizeColumns(X);
    const R = covarianceMatrix(Xs);
    const inv = safeInverseWithRidge(R);
    let communalities = R.map((row, i) => {
      const val = inv && inv[i][i] > 1e-6 ? 1 - 1 / inv[i][i] : 0.6;
      return clamp(val, 0.15, 0.98);
    });

    let loadings = Array.from({length: R.length}, () => Array(k).fill(0));
    for (let iter = 0; iter < 7; iter++) {
      const reduced = R.map((row, i) => row.map((x, j) => (i === j ? communalities[i] : x)));
      const eig = topEigenPairs(reduced, k);
      loadings = Array.from({length: R.length}, () => Array(k).fill(0));
      for (let c = 0; c < k; c++) {
        const scale = Math.sqrt(Math.max(eig.values[c], 0));
        for (let i = 0; i < R.length; i++) loadings[i][c] = eig.vectors[c][i] * scale;
      }
      communalities = loadings.map(row =>
        clamp(row.reduce((s, x) => s + x * x, 0), 0.05, 0.99)
      );
    }

    return {
      loadings,
      communalities,
      uniqueness: communalities.map(h => 1 - h)
    };
  }

  function euclideanDistance(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      const d = a[i] - b[i];
      s += d * d;
    }
    return Math.sqrt(s);
  }

  function manhattanDistance(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s;
  }

  function correlationDistance(a, b) {
    return 1 - safeCorr(a, b);
  }

  function pairwiseDistanceMatrix(X, metric) {
    const n = X.length;
    const D = Array.from({length: n}, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let d = 0;
        if (metric === "euclidean") d = euclideanDistance(X[i], X[j]);
        else if (metric === "manhattan") d = manhattanDistance(X[i], X[j]);
        else d = correlationDistance(X[i], X[j]);
        D[i][j] = d;
        D[j][i] = d;
      }
    }
    return D;
  }

  function flattenUpperTriangle(M) {
    const out = [];
    for (let i = 0; i < M.length; i++) {
      for (let j = i + 1; j < M.length; j++) out.push(M[i][j]);
    }
    return out;
  }

  function classicalMds(D) {
    const n = D.length;
    const D2 = D.map(row => row.map(x => x * x));
    const rowMeans = D2.map(mean);
    const grandMean = mean(rowMeans);
    const B = Array.from({length: n}, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        B[i][j] = -0.5 * (D2[i][j] - rowMeans[i] - rowMeans[j] + grandMean);
      }
    }
    const eig = topEigenPairs(B, 2);
    const coords = Array.from({length: n}, () => [0, 0]);
    for (let c = 0; c < 2; c++) {
      const scale = Math.sqrt(Math.max(eig.values[c], 0));
      for (let i = 0; i < n; i++) coords[i][c] = eig.vectors[c][i] * scale;
    }
    return {coords, eigenvalues: eig.values};
  }

  function coordsDistanceFit(D, coords) {
    const approx = Array.from({length: coords.length}, () => Array(coords.length).fill(0));
    for (let i = 0; i < coords.length; i++) {
      for (let j = i + 1; j < coords.length; j++) {
        const d = euclideanDistance(coords[i], coords[j]);
        approx[i][j] = d;
        approx[j][i] = d;
      }
    }
    return safeCorr(flattenUpperTriangle(D), flattenUpperTriangle(approx));
  }

  function sampleIndices(n, maxN, seed, target) {
    const all = Array.from({length: n}, (_, i) => i);
    if (n <= maxN) return all;
    const rng = mulberry32(seed);

    if (!target || target.length !== n) {
      return shuffleInPlace(all.slice(), rng).slice(0, maxN).sort((a, b) => a - b);
    }

    const groups = {};
    target.forEach((cls, i) => {
      groups[cls] = groups[cls] || [];
      groups[cls].push(i);
    });
    const keys = Object.keys(groups);
    keys.forEach(k => shuffleInPlace(groups[k], rng));
    const total = n;
    const counts = {};
    let used = 0;
    keys.forEach(k => {
      const proposed = Math.max(1, Math.floor(maxN * groups[k].length / total));
      counts[k] = Math.min(groups[k].length, proposed);
      used += counts[k];
    });
    while (used > maxN) {
      const key = keys.find(k => counts[k] > 1);
      if (!key) break;
      counts[key] -= 1;
      used -= 1;
    }
    while (used < maxN) {
      const key = keys
        .filter(k => counts[k] < groups[k].length)
        .sort((a, b) => (groups[b].length - counts[b]) - (groups[a].length - counts[a]))[0];
      if (!key) break;
      counts[key] += 1;
      used += 1;
    }

    const picked = [];
    keys.forEach(k => {
      picked.push(...groups[k].slice(0, counts[k]));
    });
    return picked.sort((a, b) => a - b);
  }

  function simulateDataset() {
    const n = Math.round(getVal("n"));
    const noise = getVal("noise");
    const nonlinear = getVal("nonlinear");
    const outliersN = Math.round(getVal("outliers"));
    const outlierStrength = getVal("outlierStrength");
    const seed = Math.round(getVal("seed"));
    const rng = mulberry32(seed);

    const X = [];
    const hidden = [];
    const specialFlags = Array(n).fill(false);
    const cols = ["x1_linear", "x2_linear", "x3_exp", "x4_tanh", "x5_cubic"];

    for (let i = 0; i < n; i++) {
      const z = randn(rng);
      hidden.push(z);
      const e = () => noise * randn(rng);
      const x1 = z + e();
      const x2 = 1.2 * z + 1.2 * e();
      const x3 = Math.exp(nonlinear * z) + e();
      const x4 = Math.tanh((0.8 + nonlinear) * z) + 0.7 * e();
      const x5 = 0.25 * nonlinear * Math.pow(z, 3) + 1.3 * e();
      X.push([x1, x2, x3, x4, x5]);
    }

    const used = new Set();
    let count = 0;
    while (count < outliersN && count < n) {
      const idx = Math.floor(rng() * n);
      if (used.has(idx)) continue;
      used.add(idx);
      specialFlags[idx] = true;
      X[idx][1] += outlierStrength + 1.2 * randn(rng);
      if (count < Math.floor(outliersN / 2)) {
        X[idx][2] += 1.6 * outlierStrength + 1.5 * randn(rng);
      }
      count++;
    }

    const q1 = quantile(hidden, 1 / 3);
    const q2 = quantile(hidden, 2 / 3);
    const target = hidden.map(z => (z < q1 ? 0 : (z < q2 ? 1 : 2)));

    return {
      mode: "simulated",
      title: "Simulated dataset",
      subtitle: "Randomly generated with one hidden driver plus noise, monotone nonlinearity, optional injected outliers, and hidden low/mid/high signal groups for the supervised section.",
      targetNote: "For simulated data, the supervised labels are hidden low/mid/high bands from the latent driver, not the outlier flag colouring.",
      X,
      columns: cols,
      specialFlags,
      target,
      target_names: ["low signal", "mid signal", "high signal"],
      n: X.length,
      p: cols.length
    };
  }

  function getRealDataset(key) {
    const ds = REAL_DATASETS[key];
    return {
      mode: "real",
      title: ds.name + " dataset",
      subtitle: "Built-in real dataset included directly inside the app.",
      targetNote: "For real datasets, the labels are the built-in dataset classes.",
      X: ds.data,
      columns: ds.columns,
      specialFlags: Array(ds.data.length).fill(false),
      target: ds.target,
      target_names: ds.target_names,
      n: ds.data.length,
      p: ds.columns.length
    };
  }

  function getActiveDataset() {
    const mode = document.getElementById("datasetMode").value;
    if (mode === "simulated") return simulateDataset();
    return getRealDataset(mode);
  }

  function colorFor(index, meta) {
    const enabled = document.getElementById("colorSpecial").checked;
    if (!enabled) return "rgba(31,119,180,0.74)";
    if (meta.mode === "simulated") {
      return meta.specialFlags[index] ? "rgba(220,38,38,0.84)" : "rgba(31,119,180,0.74)";
    }
    if (meta.target && meta.target.length === meta.n) {
      return palette[meta.target[index] % palette.length];
    }
    return "rgba(31,119,180,0.74)";
  }

  function clusterColor(label) {
    return palette[label % palette.length];
  }

  function fmt(x, d) {
    return Number(x).toFixed(d === undefined ? 3 : d);
  }

  function svgElement(name) {
    return document.createElementNS("http://www.w3.org/2000/svg", name);
  }

  function addLine(svg, x1, y1, x2, y2, stroke, width, opacity) {
    const el = svgElement("line");
    el.setAttribute("x1", x1);
    el.setAttribute("y1", y1);
    el.setAttribute("x2", x2);
    el.setAttribute("y2", y2);
    el.setAttribute("stroke", stroke);
    el.setAttribute("stroke-width", width);
    el.setAttribute("opacity", opacity === undefined ? 1 : opacity);
    svg.appendChild(el);
  }

  function addRect(svg, x, y, w, h, fill, stroke) {
    const el = svgElement("rect");
    el.setAttribute("x", x);
    el.setAttribute("y", y);
    el.setAttribute("width", w);
    el.setAttribute("height", h);
    el.setAttribute("fill", fill);
    el.setAttribute("stroke", stroke || "none");
    svg.appendChild(el);
  }

  function addCircle(svg, cx, cy, r, fill, stroke) {
    const el = svgElement("circle");
    el.setAttribute("cx", cx);
    el.setAttribute("cy", cy);
    el.setAttribute("r", r);
    el.setAttribute("fill", fill);
    el.setAttribute("stroke", stroke || "none");
    svg.appendChild(el);
  }

  function addSquare(svg, cx, cy, size, fill, stroke) {
    const el = svgElement("rect");
    el.setAttribute("x", cx - size / 2);
    el.setAttribute("y", cy - size / 2);
    el.setAttribute("width", size);
    el.setAttribute("height", size);
    el.setAttribute("fill", fill);
    el.setAttribute("stroke", stroke || "none");
    svg.appendChild(el);
  }

  function addText(svg, x, y, text, anchor, size, fill, weight) {
    const el = svgElement("text");
    el.setAttribute("x", x);
    el.setAttribute("y", y);
    el.setAttribute("text-anchor", anchor);
    el.setAttribute("font-size", size);
    el.setAttribute("fill", fill);
    if (weight) el.setAttribute("font-weight", weight);
    el.textContent = text;
    svg.appendChild(el);
  }

  function addRotatedYLabel(svg, x, y, text) {
    const el = svgElement("text");
    el.setAttribute("x", x);
    el.setAttribute("y", y);
    el.setAttribute("transform", `rotate(-90 ${x} ${y})`);
    el.setAttribute("text-anchor", "middle");
    el.setAttribute("font-size", "14");
    el.setAttribute("fill", "#111827");
    el.textContent = text;
    svg.appendChild(el);
  }

  function fmtTick(x) {
    return Math.abs(x) >= 10 ? x.toFixed(0) : x.toFixed(1);
  }

  function updateLegend(meta) {
    const legend = document.getElementById("legendRow");
    legend.innerHTML = "";
    const enabled = document.getElementById("colorSpecial").checked;
    if (!enabled) {
      legend.innerHTML = '<span><span class="swatch" style="background:rgba(31,119,180,0.74)"></span>points</span>';
      return;
    }
    if (meta.mode === "simulated") {
      legend.innerHTML =
        '<span><span class="swatch" style="background:rgba(31,119,180,0.74)"></span>regular points</span>' +
        '<span><span class="swatch" style="background:rgba(220,38,38,0.84)"></span>injected outliers</span>';
      return;
    }
    if (meta.target && meta.target_names) {
      legend.innerHTML = meta.target_names.map((name, i) =>
        `<span><span class="swatch" style="background:${palette[i % palette.length]}"></span>${name}</span>`
      ).join("");
      return;
    }
    legend.innerHTML = '<span><span class="swatch" style="background:rgba(31,119,180,0.74)"></span>points</span>';
  }

  function updateClusterLegend(k) {
    const legend = document.getElementById("clusterLegend");
    legend.innerHTML = "";
    let html = "";
    for (let i = 0; i < k; i++) {
      html += `<span><span class="swatch" style="background:${clusterColor(i)}"></span>cluster ${i + 1}</span>`;
    }
    legend.innerHTML = html;
  }

  function updateMetaPills(meta) {
    const el = document.getElementById("metaPills");
    let html = `<span class="pill">${meta.n} rows</span><span class="pill">${meta.p} variables</span>`;
    meta.columns.forEach(c => {
      html += `<span class="pill">${c}</span>`;
    });
    el.innerHTML = html;
  }

  function updateDatasetNote(meta) {
    const note = document.getElementById("datasetNote");
    if (meta.mode === "simulated") {
      note.textContent = `${meta.subtitle} ${meta.targetNote}`;
    } else {
      note.textContent = `${meta.subtitle} ${meta.targetNote} Class colouring is shown when labels are available.`;
    }
  }

  function getNodeList(selector) {
    return document.querySelectorAll ? Array.from(document.querySelectorAll(selector)) : [];
  }

  function setGroupDisabled(selector, disabled) {
    getNodeList(selector).forEach(el => {
      if (el.classList && el.classList.toggle) {
        el.classList.toggle("is-disabled", disabled);
      } else if (el.classList && disabled && el.classList.add) {
        el.classList.add("is-disabled");
      } else if (el.classList && !disabled && el.classList.remove) {
        el.classList.remove("is-disabled");
      }
      const children = el.querySelectorAll ? Array.from(el.querySelectorAll("input, select, button")) : [];
      children.forEach(ctrl => {
        ctrl.disabled = disabled;
      });
    });
  }

  function updateControlVisibility(meta) {
    const isSimulated = meta.mode === "simulated";
    const validationMode = document.getElementById("validationMode").value;
    setGroupDisabled('[data-control-group="simulated"]', !isSimulated);
    setGroupDisabled('[data-control-group="holdout"]', validationMode !== "holdout");
    setGroupDisabled('[data-control-group="cv"]', validationMode !== "cv");

    const presetButtons = ["presetLinear", "presetNonlinear", "presetOutliers"];
    presetButtons.forEach(id => {
      document.getElementById(id).disabled = !isSimulated;
    });
  }

  function updateExploreGuide(meta) {
    const datasetMode = document.getElementById("datasetMode").value;
    const distanceMetric = document.getElementById("distanceMetric").value;
    const clusterSpace = document.getElementById("clusterSpace").value;
    const validationMode = document.getElementById("validationMode").value;
    const guide = document.getElementById("exploreGuide");
    const navHint = document.getElementById("navHint");

    let guideText = "Start with PCA core, then move to Distance Explorer and MDS to see geometry, then use Clustering and Supervised Models to test whether the same structure is useful for grouping and prediction.";
    if (datasetMode === "simulated") {
      guideText += " In simulated mode, try switching between the nonlinear and outlier-heavy presets to see when rank methods become more stable than raw-value methods.";
    } else if (datasetMode === "iris") {
      guideText += " Iris is a good first real dataset because the class structure is clean enough to compare PCA, clustering, and classification side by side.";
    } else if (datasetMode === "wine") {
      guideText += " Wine is useful when you want more variables and more overlap, so the representation choices matter more.";
    } else {
      guideText += " Breast cancer is the strongest supervised-model demo here because the class boundary is fairly meaningful while still being multivariate.";
    }

    let navText = `Suggested path: PCA core -> Distance Explorer (${distanceMetricLabel(distanceMetric)}) -> MDS -> Clustering (${clusterSpace === "original" ? "original variables" : clusterSpace === "raw_pcs" ? "raw PCs" : "rank PCs"}) -> Supervised models (${validationMode === "holdout" ? "holdout" : "CV"}).`;
    if (meta.mode === "simulated") {
      navText += " The coloured points show outliers, while the supervised labels come from hidden low/mid/high latent groups.";
    }
    guide.textContent = guideText;
    navHint.textContent = navText;
  }

  function drawComparisonScatter(svgId, xVals, yVals, meta, xLabel, yLabel) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 700;
    const H = 500;
    const m = {top: 24, right: 24, bottom: 58, left: 72};

    const zx = zscore(xVals);
    const zy = zscore(yVals);
    const all = zx.concat(zy);
    const minV = Math.min(...all);
    const maxV = Math.max(...all);
    const pad = (maxV - minV) * 0.10 + 0.2;
    const xMin = minV - pad;
    const xMax = maxV + pad;
    const yMin = minV - pad;
    const yMax = maxV + pad;
    const sx = x => m.left + (x - xMin) / (xMax - xMin) * (W - m.left - m.right);
    const sy = y => H - m.bottom - (y - yMin) / (yMax - yMin) * (H - m.top - m.bottom);

    for (let i = 0; i <= 6; i++) {
      const t = xMin + i * (xMax - xMin) / 6;
      const x = sx(t);
      const y = sy(t);
      addLine(svg, x, m.top, x, H - m.bottom, "#e5e7eb", 1);
      addLine(svg, m.left, y, W - m.right, y, "#e5e7eb", 1);
      addText(svg, x, H - m.bottom + 20, fmtTick(t), "middle", 12, "#6b7280");
      addText(svg, m.left - 10, y + 4, fmtTick(t), "end", 12, "#6b7280");
    }

    addLine(svg, m.left, H - m.bottom, W - m.right, H - m.bottom, "#9ca3af", 1.2);
    addLine(svg, m.left, m.top, m.left, H - m.bottom, "#9ca3af", 1.2);
    addLine(svg, sx(xMin), sy(xMin), sx(xMax), sy(xMax), "#2563eb", 2);

    for (let i = 0; i < zx.length; i++) {
      addCircle(svg, sx(zx[i]), sy(zy[i]), 4.1, colorFor(i, meta));
    }

    addText(svg, W / 2, H - 18, xLabel, "middle", 15, "#111827");
    addRotatedYLabel(svg, 20, H / 2, yLabel);
  }

  function drawBiplot(svgId, scores1, scores2, load1, load2, meta, color) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 700;
    const H = 520;
    const m = {top: 20, right: 20, bottom: 50, left: 58};
    const showLabels = document.getElementById("showLabels").checked;
    const z1 = zscore(scores1);
    const z2 = zscore(scores2);
    const scoreMax = Math.max(...z1.map(Math.abs), ...z2.map(Math.abs));
    const lim = Math.max(2.8, scoreMax * 1.15);
    const sx = x => m.left + (x + lim) / (2 * lim) * (W - m.left - m.right);
    const sy = y => H - m.bottom - (y + lim) / (2 * lim) * (H - m.top - m.bottom);

    for (let i = 0; i <= 6; i++) {
      const t = -lim + i * (2 * lim) / 6;
      addLine(svg, sx(t), m.top, sx(t), H - m.bottom, "#eef2f7", 1);
      addLine(svg, m.left, sy(t), W - m.right, sy(t), "#eef2f7", 1);
    }

    addLine(svg, sx(-lim), sy(0), sx(lim), sy(0), "#9ca3af", 1.2);
    addLine(svg, sx(0), sy(-lim), sx(0), sy(lim), "#9ca3af", 1.2);

    for (let i = 0; i < z1.length; i++) {
      addCircle(svg, sx(z1[i]), sy(z2[i]), 3.7, colorFor(i, meta));
    }

    const arrowScale = lim * 0.72;
    for (let j = 0; j < load1.length; j++) {
      const x2 = load1[j] * arrowScale;
      const y2 = load2[j] * arrowScale;
      addLine(svg, sx(0), sy(0), sx(x2), sy(y2), color, 2.2);
      addCircle(svg, sx(x2), sy(y2), 4.5, color);
      if (showLabels) addText(svg, sx(x2) + 6, sy(y2) - 6, meta.columns[j], "start", 12, color, "bold");
    }

    addText(svg, W / 2, H - 14, "PC1 scores", "middle", 14, "#111827");
    addRotatedYLabel(svg, 18, H / 2, "PC2 scores");
  }

  function divergingColor(v) {
    const t = (v + 1) / 2;
    const r = Math.round(33 + t * 200);
    const b = Math.round(33 + (1 - t) * 200);
    const g = Math.round(50 + (1 - Math.abs(v)) * 120);
    return `rgb(${r},${g},${b})`;
  }

  function sequentialColor(v, minV, maxV) {
    const t = maxV - minV < 1e-12 ? 0 : (v - minV) / (maxV - minV);
    const r = Math.round(237 - t * 160);
    const g = Math.round(246 - t * 110);
    const b = Math.round(255 - t * 35);
    return `rgb(${r},${g},${b})`;
  }

  function drawMatrixHeatmap(svgId, M, labels, options) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 620;
    const H = 560;
    const n = M.length;
    const m = {
      top: 42,
      right: 74,
      bottom: 86,
      left: labels.length > 12 ? 64 : 84
    };
    const cellW = (W - m.left - m.right) / n;
    const cellH = (H - m.top - m.bottom) / n;
    const values = M.flat();
    const minV = options && options.min !== undefined ? options.min : Math.min(...values);
    const maxV = options && options.max !== undefined ? options.max : Math.max(...values);
    const showText = n <= 18;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = m.left + j * cellW;
        const y = m.top + i * cellH;
        const fill = options && options.mode === "diverging"
          ? divergingColor(M[i][j])
          : sequentialColor(M[i][j], minV, maxV);
        addRect(svg, x, y, cellW, cellH, fill, "#ffffff");
        if (showText) addText(svg, x + cellW / 2, y + cellH / 2 + 4, fmt(M[i][j], 2), "middle", 11, "#111827");
      }
    }

    for (let i = 0; i < n; i++) {
      const label = labels[i];
      addText(svg, m.left + i * cellW + cellW / 2, H - m.bottom + 18, label, "middle", 10, "#111827");
      addText(svg, m.left - 8, m.top + i * cellH + cellH / 2 + 4, label, "end", 10, "#111827");
    }

    const lx = W - 40;
    const ly = m.top;
    const lh = n * cellH;
    for (let k = 0; k < 100; k++) {
      const t = k / 99;
      const v = options && options.mode === "diverging"
        ? (1 - 2 * t)
        : (maxV - (maxV - minV) * t);
      const fill = options && options.mode === "diverging"
        ? divergingColor(v)
        : sequentialColor(v, minV, maxV);
      addRect(svg, lx, ly + k * (lh / 100), 16, lh / 100 + 1, fill);
    }
    addText(svg, lx + 22, ly + 4, fmt(maxV, options && options.mode === "diverging" ? 1 : 2), "start", 11, "#111827");
    addText(svg, lx + 22, ly + lh / 2 + 4, fmt((minV + maxV) / 2, options && options.mode === "diverging" ? 1 : 2), "start", 11, "#111827");
    addText(svg, lx + 22, ly + lh, fmt(minV, options && options.mode === "diverging" ? 1 : 2), "start", 11, "#111827");
  }

  function drawProjectionScatter(svgId, coords, colors, xLabel, yLabel, pointLabels, shapes) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 700;
    const H = 500;
    const m = {top: 22, right: 24, bottom: 54, left: 62};
    const xs = coords.map(p => p[0]);
    const ys = coords.map(p => p[1]);
    const xMin0 = Math.min(...xs);
    const xMax0 = Math.max(...xs);
    const yMin0 = Math.min(...ys);
    const yMax0 = Math.max(...ys);
    const xPad = (xMax0 - xMin0) * 0.15 + 0.2;
    const yPad = (yMax0 - yMin0) * 0.15 + 0.2;
    const xMin = xMin0 - xPad;
    const xMax = xMax0 + xPad;
    const yMin = yMin0 - yPad;
    const yMax = yMax0 + yPad;
    const sx = x => m.left + (x - xMin) / (xMax - xMin || 1) * (W - m.left - m.right);
    const sy = y => H - m.bottom - (y - yMin) / (yMax - yMin || 1) * (H - m.top - m.bottom);

    for (let i = 0; i <= 6; i++) {
      const tx = xMin + i * (xMax - xMin) / 6;
      const ty = yMin + i * (yMax - yMin) / 6;
      addLine(svg, sx(tx), m.top, sx(tx), H - m.bottom, "#eef2f7", 1);
      addLine(svg, m.left, sy(ty), W - m.right, sy(ty), "#eef2f7", 1);
      addText(svg, sx(tx), H - m.bottom + 18, fmtTick(tx), "middle", 11, "#6b7280");
      addText(svg, m.left - 8, sy(ty) + 4, fmtTick(ty), "end", 11, "#6b7280");
    }

    if (xMin <= 0 && xMax >= 0) addLine(svg, sx(0), m.top, sx(0), H - m.bottom, "#9ca3af", 1.1);
    if (yMin <= 0 && yMax >= 0) addLine(svg, m.left, sy(0), W - m.right, sy(0), "#9ca3af", 1.1);
    addLine(svg, m.left, H - m.bottom, W - m.right, H - m.bottom, "#9ca3af", 1.1);
    addLine(svg, m.left, m.top, m.left, H - m.bottom, "#9ca3af", 1.1);

    coords.forEach((point, i) => {
      const px = sx(point[0]);
      const py = sy(point[1]);
      const shape = shapes ? shapes[i] : "circle";
      if (shape === "square") addSquare(svg, px, py, 9, colors[i], "#ffffff");
      else addCircle(svg, px, py, 4.4, colors[i], "#ffffff");
      if (pointLabels && pointLabels[i]) addText(svg, px + 6, py - 6, pointLabels[i], "start", 11, "#111827", "bold");
    });

    addText(svg, W / 2, H - 16, xLabel, "middle", 14, "#111827");
    addRotatedYLabel(svg, 18, H / 2, yLabel);
  }

  function drawLoadingMap(svgId, loadings, labels, color) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 700;
    const H = 500;
    const m = {top: 24, right: 24, bottom: 54, left: 62};
    const lim = 1.25;
    const sx = x => m.left + (x + lim) / (2 * lim) * (W - m.left - m.right);
    const sy = y => H - m.bottom - (y + lim) / (2 * lim) * (H - m.top - m.bottom);
    for (let i = 0; i <= 6; i++) {
      const t = -lim + i * (2 * lim) / 6;
      addLine(svg, sx(t), m.top, sx(t), H - m.bottom, "#eef2f7", 1);
      addLine(svg, m.left, sy(t), W - m.right, sy(t), "#eef2f7", 1);
    }
    addLine(svg, sx(-lim), sy(0), sx(lim), sy(0), "#9ca3af", 1.2);
    addLine(svg, sx(0), sy(-lim), sx(0), sy(lim), "#9ca3af", 1.2);
    loadings.forEach((row, i) => {
      addLine(svg, sx(0), sy(0), sx(row[0]), sy(row[1]), color, 2);
      addCircle(svg, sx(row[0]), sy(row[1]), 4.6, color);
      addText(svg, sx(row[0]) + 6, sy(row[1]) - 6, labels[i], "start", 11, color, "bold");
    });
    addText(svg, W / 2, H - 16, "Dimension 1", "middle", 14, "#111827");
    addRotatedYLabel(svg, 18, H / 2, "Dimension 2");
  }

  function updateLoadingsTable(raw, rank, columns) {
    const tbody = document.querySelector("#loadingsTable tbody");
    tbody.innerHTML = "";
    for (let i = 0; i < columns.length; i++) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${columns[i]}</td>
        <td>${fmt(raw.components[0][i])}</td>
        <td>${fmt(raw.components[1][i])}</td>
        <td>${fmt(rank.components[0][i])}</td>
        <td>${fmt(rank.components[1][i])}</td>
      `;
      tbody.appendChild(tr);
    }
  }

  function distanceMetricLabel(metric) {
    if (metric === "euclidean") return "Euclidean";
    if (metric === "manhattan") return "Manhattan";
    if (metric === "pearson") return "Pearson distance";
    return "Spearman-style distance";
  }

  function computePairLabelNote(meta, indices, rawMatrix, rankMatrix) {
    const target = meta.target ? subsetVector(meta.target, indices) : null;
    if (target && target.length === indices.length) {
      let rawSame = [];
      let rawDiff = [];
      let rankSame = [];
      let rankDiff = [];
      for (let i = 0; i < target.length; i++) {
        for (let j = i + 1; j < target.length; j++) {
          if (target[i] === target[j]) {
            rawSame.push(rawMatrix[i][j]);
            rankSame.push(rankMatrix[i][j]);
          } else {
            rawDiff.push(rawMatrix[i][j]);
            rankDiff.push(rankMatrix[i][j]);
          }
        }
      }
      const rawGap = mean(rawDiff) - mean(rawSame);
      const rankGap = mean(rankDiff) - mean(rankSame);
      if (rankGap > rawGap + 0.02) return "ranking separates labelled groups more";
      if (rawGap > rankGap + 0.02) return "raw values separate labelled groups more";
      return "group separation is similar";
    }
    return "compare local neighbourhood changes";
  }

  function updateDistanceSection(meta, rawStandard, rankStandard, heatIndices) {
    const metric = document.getElementById("distanceMetric").value;
    const rawSubset = subsetRows(rawStandard, heatIndices);
    const rankSubset = subsetRows(rankStandard, heatIndices);
    const rawMetric = metric === "manhattan" ? "manhattan" : (metric === "euclidean" ? "euclidean" : "correlation");
    const rankMetric = metric === "spearman" || metric === "pearson"
      ? "correlation"
      : rawMetric;
    const rawMatrix = pairwiseDistanceMatrix(rawSubset, rawMetric);
    const rankMatrix = pairwiseDistanceMatrix(rankSubset, rankMetric);
    const labels = heatIndices.map(i => String(i + 1));
    const flatRaw = flattenUpperTriangle(rawMatrix);
    const flatRank = flattenUpperTriangle(rankMatrix);
    const change = flatRaw.map((x, i) => Math.abs(x - flatRank[i]));
    const distanceCorr = safeCorr(flatRaw, flatRank);
    const note = computePairLabelNote(meta, heatIndices, rawMatrix, rankMatrix);

    document.getElementById("distanceMetricVal").textContent = distanceMetricLabel(metric);
    document.getElementById("distanceCorrVal").textContent = fmt(distanceCorr);
    document.getElementById("distanceShiftVal").textContent = fmt(mean(change));
    document.getElementById("distanceClassVal").textContent = note;
    document.getElementById("distanceRawTitle").textContent =
      metric === "pearson"
        ? "Pearson / correlation-distance heatmap"
        : metric === "spearman"
          ? "Raw Pearson-distance heatmap"
          : "Raw-value distance heatmap";
    document.getElementById("distanceRankTitle").textContent =
      metric === "pearson"
        ? "Ranked-data correlation-distance heatmap"
        : metric === "spearman"
          ? "Rank / Spearman-distance heatmap"
          : "Rank-transformed distance heatmap";
    document.getElementById("distanceRawCaption").textContent =
      metric === "manhattan"
        ? "Scaled raw-variable Manhattan distances on the sampled observations."
        : metric === "euclidean"
          ? "Scaled raw-variable Euclidean distances on the sampled observations."
          : metric === "pearson"
            ? "Correlation distance uses 1 - Pearson correlation between raw observation profiles."
            : "This left panel keeps the raw Pearson-distance view so you can compare it directly against the Spearman-style version.";
    document.getElementById("distanceRankCaption").textContent =
      metric === "manhattan"
        ? "The same Manhattan idea after ranking each variable first."
        : metric === "euclidean"
          ? "The same Euclidean idea after ranking each variable first."
          : metric === "pearson"
            ? "This keeps Pearson-style correlation distance but after ranking the variables first, so spacing changes but the distance formula does not."
            : "Spearman-style distance is Pearson distance after ranking variables first.";

    drawMatrixHeatmap("distanceHeatRaw", rawMatrix, labels, {mode: "sequential", min: 0});
    drawMatrixHeatmap("distanceHeatRank", rankMatrix, labels, {mode: "sequential", min: 0});

    let summary = "";
    if (distanceCorr > 0.9) {
      summary = "Ranking leaves most pairwise structure intact. The broad geometry is similar before and after the rank transform.";
    } else if (distanceCorr > 0.7) {
      summary = "Ranking keeps the large-scale pattern but clearly reorders some neighbourhoods. This is where raw spacing and pure ordering start telling different stories.";
    } else {
      summary = "Ranking changes the geometry substantially. Observations that were far apart in raw units are no longer far apart once only order information is retained.";
    }
    if (meta.mode === "simulated") {
      summary += " In the simulated data, that usually means outliers and nonlinear stretches matter less after ranking.";
    } else {
      summary += " On the real data, use this to see whether raw magnitudes or monotone ordering carry more of the structure.";
    }
    document.getElementById("distanceExplain").textContent =
      `${summary} Sample size here is ${heatIndices.length} observations so the distance matrix stays readable.`;
  }

  function updateMdsSection(meta, rawPca, rankPca, rawStandard, rankStandard, analysisIndices) {
    const rawSubset = subsetRows(rawStandard, analysisIndices);
    const rankSubset = subsetRows(rankStandard, analysisIndices);
    const rawD = pairwiseDistanceMatrix(rawSubset, "euclidean");
    const rankD = pairwiseDistanceMatrix(rankSubset, "correlation");
    const rawMds = classicalMds(rawD);
    const rankMds = classicalMds(rankD);
    const rawPcaCoords = analysisIndices.map(i => [rawPca.scores[0][i], rawPca.scores[1][i]]);
    const rankPcaCoords = analysisIndices.map(i => [rankPca.scores[0][i], rankPca.scores[1][i]]);
    const rawFit = coordsDistanceFit(rawD, rawMds.coords);
    const rawPcaFit = coordsDistanceFit(rawD, rawPcaCoords);
    const rankFit = coordsDistanceFit(rankD, rankMds.coords);
    const rankPcaFit = coordsDistanceFit(rankD, rankPcaCoords);
    const colors = analysisIndices.map(i => colorFor(i, meta));

    drawProjectionScatter("mdsRaw", rawMds.coords, colors, "MDS axis 1", "MDS axis 2");
    drawProjectionScatter("mdsRank", rankMds.coords, colors, "MDS axis 1", "MDS axis 2");

    document.getElementById("mdsRawFitVal").textContent = fmt(rawFit);
    document.getElementById("mdsRawPcaFitVal").textContent = fmt(rawPcaFit);
    document.getElementById("mdsRankFitVal").textContent = fmt(rankFit);
    document.getElementById("mdsRankPcaFitVal").textContent = fmt(rankPcaFit);
    document.getElementById("mdsRawCaption").textContent =
      `Classical MDS on Euclidean distances between scaled raw observations. ${analysisIndices.length < meta.n ? "A reproducible subset is used here." : ""}`;
    document.getElementById("mdsRankCaption").textContent =
      "Classical MDS on Spearman-style correlation distances after ranking each variable first.";

    const betterRaw = rawFit - rawPcaFit;
    const betterRank = rankFit - rankPcaFit;
    let text = "MDS works from distances directly, so it often preserves pairwise geometry a bit better than PCA when the target notion of similarity is distance-based rather than variance-based.";
    if (betterRaw > 0.03) text += ` For the raw-distance view, MDS is noticeably closer to the original distances than the first two raw PCs (${fmt(rawFit)} vs ${fmt(rawPcaFit)}).`;
    else text += ` For the raw-distance view, MDS and PCA are giving a fairly similar two-dimensional geometry (${fmt(rawFit)} vs ${fmt(rawPcaFit)}).`;
    if (betterRank > 0.03) text += ` The rank/correlation-distance view especially benefits from MDS here (${fmt(rankFit)} vs ${fmt(rankPcaFit)}).`;
    else text += ` The rank-distance MDS and rank PCA views are telling a similar story in two dimensions.`;
    document.getElementById("mdsExplain").textContent = text;
  }

  function nearestCenterIndex(row, centers) {
    let best = 0;
    let bestDist = Infinity;
    centers.forEach((center, i) => {
      const d = euclideanDistance(row, center);
      if (d < bestDist) {
        best = i;
        bestDist = d;
      }
    });
    return best;
  }

  function runKMeans(X, k, seed) {
    const rng = mulberry32(seed);
    const centers = [];
    const first = Math.floor(rng() * X.length);
    centers.push(X[first].slice());
    while (centers.length < k) {
      const d2 = X.map(row => {
        const nearest = centers.reduce((best, c) => Math.min(best, Math.pow(euclideanDistance(row, c), 2)), Infinity);
        return nearest;
      });
      const total = sum(d2) || 1;
      let r = rng() * total;
      let pick = 0;
      for (let i = 0; i < d2.length; i++) {
        r -= d2[i];
        if (r <= 0) {
          pick = i;
          break;
        }
      }
      centers.push(X[pick].slice());
    }

    let labels = Array(X.length).fill(0);
    for (let iter = 0; iter < 40; iter++) {
      const nextLabels = X.map(row => nearestCenterIndex(row, centers));
      let changed = false;
      for (let i = 0; i < labels.length; i++) if (labels[i] !== nextLabels[i]) changed = true;
      labels = nextLabels;
      const groups = Array.from({length: k}, () => []);
      labels.forEach((lab, i) => groups[lab].push(X[i]));
      for (let c = 0; c < k; c++) {
        if (!groups[c].length) {
          centers[c] = X[Math.floor(rng() * X.length)].slice();
          continue;
        }
        centers[c] = centers[c].map((_, j) => mean(groups[c].map(row => row[j])));
      }
      if (!changed) break;
    }

    return {labels, centers};
  }

  function runHierarchical(X, k) {
    const D = pairwiseDistanceMatrix(X, "euclidean");
    let clusters = X.map((_, i) => [i]);
    while (clusters.length > k) {
      let bestI = 0;
      let bestJ = 1;
      let bestDist = Infinity;
      for (let i = 0; i < clusters.length; i++) {
        for (let j = i + 1; j < clusters.length; j++) {
          let total = 0;
          let count = 0;
          clusters[i].forEach(a => {
            clusters[j].forEach(b => {
              total += D[a][b];
              count += 1;
            });
          });
          const avg = total / Math.max(1, count);
          if (avg < bestDist) {
            bestDist = avg;
            bestI = i;
            bestJ = j;
          }
        }
      }
      clusters[bestI] = clusters[bestI].concat(clusters[bestJ]);
      clusters.splice(bestJ, 1);
    }
    const labels = Array(X.length).fill(0);
    clusters.forEach((cluster, idx) => cluster.forEach(i => { labels[i] = idx; }));
    return {labels};
  }

  function computeClusterCentroids(X, labels, k) {
    return Array.from({length: k}, (_, c) => {
      const rows = X.filter((_, i) => labels[i] === c);
      if (!rows.length) return Array(X[0].length).fill(0);
      return Array.from({length: X[0].length}, (_, j) => mean(rows.map(row => row[j])));
    });
  }

  function computeWcss(X, labels, k) {
    const centers = computeClusterCentroids(X, labels, k);
    let total = 0;
    X.forEach((row, i) => {
      const center = centers[labels[i]];
      const d = euclideanDistance(row, center);
      total += d * d;
    });
    return total;
  }

  function silhouetteScore(X, labels, k) {
    const D = pairwiseDistanceMatrix(X, "euclidean");
    const groups = Array.from({length: k}, () => []);
    labels.forEach((lab, i) => groups[lab].push(i));
    const sil = labels.map((lab, i) => {
      const same = groups[lab].filter(idx => idx !== i);
      const a = same.length ? mean(same.map(j => D[i][j])) : 0;
      let b = Infinity;
      for (let other = 0; other < k; other++) {
        if (other === lab || !groups[other].length) continue;
        const candidate = mean(groups[other].map(j => D[i][j]));
        if (candidate < b) b = candidate;
      }
      if (!isFinite(b) && a === 0) return 0;
      return (b - a) / Math.max(a, b, 1e-9);
    });
    return mean(sil);
  }

  function comb2(n) {
    return n < 2 ? 0 : n * (n - 1) / 2;
  }

  function adjustedRandIndex(trueLabels, predLabels) {
    const contingency = {};
    const a = {};
    const b = {};
    for (let i = 0; i < trueLabels.length; i++) {
      const key = `${trueLabels[i]}|${predLabels[i]}`;
      contingency[key] = (contingency[key] || 0) + 1;
      a[trueLabels[i]] = (a[trueLabels[i]] || 0) + 1;
      b[predLabels[i]] = (b[predLabels[i]] || 0) + 1;
    }
    const index = sum(Object.values(contingency).map(comb2));
    const aSum = sum(Object.values(a).map(comb2));
    const bSum = sum(Object.values(b).map(comb2));
    const total = comb2(trueLabels.length) || 1;
    const expected = aSum * bSum / total;
    const maxIndex = 0.5 * (aSum + bSum);
    const denom = maxIndex - expected;
    return denom === 0 ? 0 : (index - expected) / denom;
  }

  function majorityLabel(target, names) {
    if (!target || !target.length) return "—";
    const counts = {};
    target.forEach(t => { counts[t] = (counts[t] || 0) + 1; });
    const best = Object.keys(counts).sort((a, b) => counts[b] - counts[a])[0];
    return names && names[best] ? names[best] : String(best);
  }

  function updateClusterSection(meta, rawPca, rankPca, rawStandard, rankStandard, analysisIndices) {
    const k = Math.round(getVal("clusterK"));
    const method = document.getElementById("clusterMethod").value;
    const space = document.getElementById("clusterSpace").value;
    const pcCount = Math.min(Math.round(getVal("pcCount")), meta.p);

    let features;
    if (space === "original") {
      features = subsetRows(rawStandard, analysisIndices);
    } else if (space === "raw_pcs") {
      features = analysisIndices.map(i => rawPca.scores.slice(0, pcCount).map(score => score[i]));
    } else {
      features = analysisIndices.map(i => rankPca.scores.slice(0, pcCount).map(score => score[i]));
    }

    const model = method === "kmeans"
      ? runKMeans(features, k, Math.round(getVal("analysisSeed")))
      : runHierarchical(features, k);
    const labels = model.labels;
    const colors = labels.map(clusterColor);
    const rawCoords = analysisIndices.map(i => [rawPca.scores[0][i], rawPca.scores[1][i]]);
    const rankCoords = analysisIndices.map(i => [rankPca.scores[0][i], rankPca.scores[1][i]]);
    const silhouette = silhouetteScore(features, labels, k);
    const sse = computeWcss(features, labels, k);
    const subsetTarget = meta.target ? subsetVector(meta.target, analysisIndices) : null;
    const ari = subsetTarget ? adjustedRandIndex(subsetTarget, labels) : 0;

    drawProjectionScatter("clusterRawPlot", rawCoords, colors, "raw PC1", "raw PC2");
    drawProjectionScatter("clusterRankPlot", rankCoords, colors, "rank PC1", "rank PC2");
    updateClusterLegend(k);

    document.getElementById("clusterMethodVal").textContent =
      `${method === "kmeans" ? "K-means" : "Hierarchical"} on ${space === "original" ? "original variables" : (space === "raw_pcs" ? `${pcCount} raw PCs` : `${pcCount} rank PCs`)}`;
    document.getElementById("clusterSilhouetteVal").textContent = fmt(silhouette);
    document.getElementById("clusterSseVal").textContent = fmt(sse, 2);
    document.getElementById("clusterAriVal").textContent = meta.target ? fmt(ari) : "—";
    document.getElementById("clusterRawCaption").textContent =
      "The colours are the cluster assignments, but the coordinates are still the raw PCA projection.";
    document.getElementById("clusterRankCaption").textContent =
      "The same cluster labels, now viewed in rank PCA space. If the colours break apart differently here, the grouping depends on representation.";

    const tbody = document.querySelector("#clusterTable tbody");
    tbody.innerHTML = "";
    for (let c = 0; c < k; c++) {
      const idx = labels.map((lab, i) => (lab === c ? i : -1)).filter(i => i >= 0);
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>Cluster ${c + 1}</td>
        <td>${idx.length}</td>
        <td>${subsetTarget ? majorityLabel(idx.map(i => subsetTarget[i]), meta.target_names) : "—"}</td>
      `;
      tbody.appendChild(tr);
    }

    let explain = `${method === "kmeans" ? "K-means" : "Hierarchical clustering"} is run on a reproducible sample of ${analysisIndices.length} observations so the browser stays responsive.`;
    if (silhouette > 0.35) explain += " The silhouette score suggests moderately separated groups.";
    else explain += " The silhouette score suggests overlapping groups, so treat the cluster boundaries as soft rather than absolute.";
    if (meta.target) explain += ` The adjusted Rand index (${fmt(ari)}) shows how closely the unsupervised clusters line up with the known labels.`;
    explain += " Model-based clustering is left for later so this version stays focused and lightweight.";
    document.getElementById("clusterExplain").textContent = explain;
  }

  function uniqueValues(arr) {
    return Array.from(new Set(arr)).sort((a, b) => a - b);
  }

  function evaluatePredictions(yTrue, yPred) {
    const classes = uniqueValues(yTrue.concat(yPred));
    const accuracy = mean(yTrue.map((y, i) => (y === yPred[i] ? 1 : 0)));
    const f1s = classes.map(cls => {
      let tp = 0;
      let fp = 0;
      let fn = 0;
      for (let i = 0; i < yTrue.length; i++) {
        if (yTrue[i] === cls && yPred[i] === cls) tp += 1;
        if (yTrue[i] !== cls && yPred[i] === cls) fp += 1;
        if (yTrue[i] === cls && yPred[i] !== cls) fn += 1;
      }
      const precision = tp / Math.max(1, tp + fp);
      const recall = tp / Math.max(1, tp + fn);
      return precision + recall === 0 ? 0 : 2 * precision * recall / (precision + recall);
    });
    return {accuracy, macroF1: mean(f1s)};
  }

  function stratifiedHoldout(y, fraction, seed) {
    const rng = mulberry32(seed);
    const groups = {};
    y.forEach((cls, i) => {
      groups[cls] = groups[cls] || [];
      groups[cls].push(i);
    });
    const train = [];
    const test = [];
    Object.values(groups).forEach(indices => {
      shuffleInPlace(indices, rng);
      let testCount = Math.max(1, Math.round(indices.length * fraction));
      if (indices.length - testCount < 1) testCount = Math.max(1, indices.length - 1);
      test.push(...indices.slice(0, testCount));
      train.push(...indices.slice(testCount));
    });
    return {train, test};
  }

  function stratifiedFolds(y, folds, seed) {
    const rng = mulberry32(seed);
    const out = Array.from({length: folds}, () => []);
    const groups = {};
    y.forEach((cls, i) => {
      groups[cls] = groups[cls] || [];
      groups[cls].push(i);
    });
    Object.values(groups).forEach(indices => {
      shuffleInPlace(indices, rng);
      indices.forEach((idx, order) => {
        out[order % folds].push(idx);
      });
    });
    return out.map(fold => fold.sort((a, b) => a - b));
  }

  function buildRepresentation(trainX, testX, repr, k) {
    if (repr === "original") {
      const scaler = fitScaler(trainX);
      return {
        train: transformScaler(trainX, scaler),
        test: transformScaler(testX, scaler)
      };
    }

    if (repr === "raw_pcs") {
      const scaler = fitScaler(trainX);
      const trainScaled = transformScaler(trainX, scaler);
      const testScaled = transformScaler(testX, scaler);
      const pcaObj = pcaFromStandardized(trainScaled, k);
      return {
        train: projectRows(trainScaled, pcaObj.components, k),
        test: projectRows(testScaled, pcaObj.components, k)
      };
    }

    const trainRank = rankColumnsNormalized(trainX);
    const rankRef = fitRankReference(trainX);
    const testRank = transformRankReference(testX, rankRef);
    const scaler = fitScaler(trainRank);
    const trainScaled = transformScaler(trainRank, scaler);
    const testScaled = transformScaler(testRank, scaler);
    const pcaObj = pcaFromStandardized(trainScaled, k);
    return {
      train: projectRows(trainScaled, pcaObj.components, k),
      test: projectRows(testScaled, pcaObj.components, k)
    };
  }

  function trainKnn(trainX, trainY) {
    return {trainX, trainY};
  }

  function predictKnn(model, testX, k) {
    return testX.map(row => {
      const neighbours = model.trainX.map((trainRow, i) => ({
        label: model.trainY[i],
        dist: euclideanDistance(row, trainRow)
      })).sort((a, b) => a.dist - b.dist).slice(0, k);
      const votes = {};
      neighbours.forEach(n => { votes[n.label] = (votes[n.label] || 0) + 1; });
      return Number(Object.keys(votes).sort((a, b) => votes[b] - votes[a])[0]);
    });
  }

  function trainSoftmax(trainX, trainY, seed) {
    const rng = mulberry32(seed);
    const classes = uniqueValues(trainY);
    const C = classes.length;
    const d = trainX[0].length;
    const W = Array.from({length: C}, () =>
      Array.from({length: d + 1}, () => (rng() - 0.5) * 0.04)
    );
    const lr = 0.22 / Math.sqrt(d);
    const reg = 0.01;

    for (let iter = 0; iter < 360; iter++) {
      const grad = Array.from({length: C}, () => Array(d + 1).fill(0));
      for (let i = 0; i < trainX.length; i++) {
        const xb = trainX[i].concat(1);
        const scores = W.map(row => dot(row, xb));
        const maxScore = Math.max(...scores);
        const exps = scores.map(s => Math.exp(s - maxScore));
        const denom = sum(exps) || 1;
        const probs = exps.map(e => e / denom);
        for (let c = 0; c < C; c++) {
          const diff = probs[c] - (trainY[i] === classes[c] ? 1 : 0);
          for (let j = 0; j < d + 1; j++) grad[c][j] += diff * xb[j] / trainX.length;
        }
      }
      for (let c = 0; c < C; c++) {
        for (let j = 0; j < d + 1; j++) {
          W[c][j] -= lr * (grad[c][j] + reg * (j === d ? 0 : W[c][j]));
        }
      }
    }
    return {W, classes};
  }

  function predictSoftmax(model, testX) {
    return testX.map(row => {
      const xb = row.concat(1);
      const scores = model.W.map(w => dot(w, xb));
      let best = 0;
      for (let c = 1; c < scores.length; c++) if (scores[c] > scores[best]) best = c;
      return model.classes[best];
    });
  }

  function trainLda(trainX, trainY) {
    const classes = uniqueValues(trainY);
    const means = {};
    const priors = {};
    classes.forEach(cls => {
      const rows = trainX.filter((_, i) => trainY[i] === cls);
      priors[cls] = rows.length / trainX.length;
      means[cls] = Array.from({length: trainX[0].length}, (_, j) => mean(rows.map(row => row[j])));
    });
    const cov = Array.from({length: trainX[0].length}, () => Array(trainX[0].length).fill(0));
    trainX.forEach((row, i) => {
      const mu = means[trainY[i]];
      for (let a = 0; a < row.length; a++) {
        for (let b = 0; b < row.length; b++) {
          cov[a][b] += (row[a] - mu[a]) * (row[b] - mu[b]);
        }
      }
    });
    for (let a = 0; a < cov.length; a++) {
      for (let b = 0; b < cov.length; b++) cov[a][b] /= Math.max(1, trainX.length - classes.length);
      cov[a][a] += 0.05;
    }
    const inv = safeInverseWithRidge(cov);
    return {classes, means, priors, inv};
  }

  function predictLda(model, testX) {
    return testX.map(row => {
      let best = model.classes[0];
      let bestScore = -Infinity;
      model.classes.forEach(cls => {
        const mu = model.means[cls];
        const invMu = model.inv.map(r => dot(r, mu));
        const score = dot(row, invMu) - 0.5 * dot(mu, invMu) + Math.log(model.priors[cls] || 1e-9);
        if (score > bestScore) {
          bestScore = score;
          best = cls;
        }
      });
      return best;
    });
  }

  function gini(labels) {
    const counts = {};
    labels.forEach(y => { counts[y] = (counts[y] || 0) + 1; });
    const total = labels.length || 1;
    return 1 - sum(Object.values(counts).map(c => Math.pow(c / total, 2)));
  }

  function buildDecisionTree(X, y, depth, maxDepth, minLeaf, importances) {
    const node = {
      prediction: majorityLabel(y),
      feature: null,
      threshold: null,
      left: null,
      right: null
    };
    const parentImpurity = gini(y);
    if (depth >= maxDepth || parentImpurity < 1e-8 || X.length < 2 * minLeaf) return node;

    let best = null;
    for (let j = 0; j < X[0].length; j++) {
      const values = X.map(row => row[j]).sort((a, b) => a - b);
      const unique = Array.from(new Set(values));
      if (unique.length < 2) continue;
      const thresholds = [];
      if (unique.length <= 16) {
        for (let i = 0; i < unique.length - 1; i++) thresholds.push((unique[i] + unique[i + 1]) / 2);
      } else {
        for (let q = 1; q <= 15; q++) thresholds.push(quantile(unique, q / 16));
      }
      thresholds.forEach(threshold => {
        const leftIdx = [];
        const rightIdx = [];
        X.forEach((row, i) => {
          if (row[j] <= threshold) leftIdx.push(i);
          else rightIdx.push(i);
        });
        if (leftIdx.length < minLeaf || rightIdx.length < minLeaf) return;
        const leftY = leftIdx.map(i => y[i]);
        const rightY = rightIdx.map(i => y[i]);
        const score = (leftIdx.length / X.length) * gini(leftY) + (rightIdx.length / X.length) * gini(rightY);
        const gain = parentImpurity - score;
        if (!best || gain > best.gain) {
          best = {feature: j, threshold, leftIdx, rightIdx, gain};
        }
      });
    }

    if (!best || best.gain < 1e-6) return node;
    importances[best.feature] = (importances[best.feature] || 0) + best.gain * X.length;
    node.feature = best.feature;
    node.threshold = best.threshold;
    node.left = buildDecisionTree(best.leftIdx.map(i => X[i]), best.leftIdx.map(i => y[i]), depth + 1, maxDepth, minLeaf, importances);
    node.right = buildDecisionTree(best.rightIdx.map(i => X[i]), best.rightIdx.map(i => y[i]), depth + 1, maxDepth, minLeaf, importances);
    return node;
  }

  function trainDecisionTree(trainX, trainY) {
    const importances = {};
    const tree = buildDecisionTree(trainX, trainY, 0, 3, 5, importances);
    return {tree, importances};
  }

  function predictDecisionTree(model, testX) {
    return testX.map(row => {
      let node = model.tree;
      while (node.feature !== null) node = row[node.feature] <= node.threshold ? node.left : node.right;
      return Number(node.prediction);
    });
  }

  function runClassifier(name, trainX, trainY, testX, seed) {
    if (name === "KNN") return predictKnn(trainKnn(trainX, trainY), testX, 5);
    if (name === "Logistic regression") return predictSoftmax(trainSoftmax(trainX, trainY, seed), testX);
    if (name === "LDA") return predictLda(trainLda(trainX, trainY), testX);
    return predictDecisionTree(trainDecisionTree(trainX, trainY), testX);
  }

  function evaluateModelRepresentation(X, y, repr, modelName, config) {
    const k = config.pcCount;
    if (config.mode === "holdout") {
      const split = stratifiedHoldout(y, config.testFraction, config.seed + repr.length + modelName.length);
      const trainX = split.train.map(i => X[i]);
      const testX = split.test.map(i => X[i]);
      const trainY = split.train.map(i => y[i]);
      const testY = split.test.map(i => y[i]);
      const rep = buildRepresentation(trainX, testX, repr, k);
      const pred = runClassifier(modelName, rep.train, trainY, rep.test, config.seed + 17);
      return evaluatePredictions(testY, pred);
    }

    const folds = stratifiedFolds(y, config.folds, config.seed + repr.length + modelName.length);
    const preds = Array(y.length).fill(y[0]);
    folds.forEach((testIdx, fold) => {
      const testSet = new Set(testIdx);
      const trainIdx = y.map((_, i) => i).filter(i => !testSet.has(i));
      const trainX = trainIdx.map(i => X[i]);
      const trainY = trainIdx.map(i => y[i]);
      const testX = testIdx.map(i => X[i]);
      const rep = buildRepresentation(trainX, testX, repr, k);
      const pred = runClassifier(modelName, rep.train, trainY, rep.test, config.seed + 31 + fold);
      testIdx.forEach((originalIndex, i) => { preds[originalIndex] = pred[i]; });
    });
    return evaluatePredictions(y, preds);
  }

  function normaliseImportances(countMap, p) {
    const arr = Array.from({length: p}, (_, i) => countMap[i] || 0);
    const total = sum(arr) || 1;
    return arr.map(v => v / total);
  }

  function updateSupervisedSection(meta) {
    const pcCount = Math.min(Math.round(getVal("pcCount")), meta.p);
    document.getElementById("pcCountInline").textContent = String(pcCount);
    document.getElementById("pcCountInline2").textContent = String(pcCount);

    const y = meta.target;
    const X = meta.X;
    const models = ["KNN", "Logistic regression", "LDA", "Decision tree"];
    const reps = [
      {key: "original", label: "Original variables"},
      {key: "raw_pcs", label: `${pcCount} raw PCs`},
      {key: "rank_pcs", label: `${pcCount} rank PCs`}
    ];
    const config = {
      mode: document.getElementById("validationMode").value,
      testFraction: getVal("testFraction"),
      folds: Math.round(getVal("cvFolds")),
      seed: Math.round(getVal("analysisSeed")),
      pcCount
    };

    const results = [];
    reps.forEach(rep => {
      models.forEach(model => {
        const metrics = evaluateModelRepresentation(X, y, rep.key, model, config);
        results.push({
          model,
          representation: rep.label,
          accuracy: metrics.accuracy,
          macroF1: metrics.macroF1
        });
      });
    });

    results.sort((a, b) => b.accuracy - a.accuracy || b.macroF1 - a.macroF1);
    const best = results[0];
    document.getElementById("supervisedBestAccVal").textContent = fmt(best.accuracy);
    document.getElementById("supervisedBestF1Val").textContent = fmt(best.macroF1);
    document.getElementById("supervisedValidationVal").textContent =
      config.mode === "holdout"
        ? `${Math.round((1 - config.testFraction) * 100)}/${Math.round(config.testFraction * 100)} split`
        : `${config.folds}-fold CV`;
    document.getElementById("supervisedBestModelVal").textContent =
      `${best.model} on ${best.representation}`;

    const tbody = document.querySelector("#supervisedTable tbody");
    tbody.innerHTML = "";
    results.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.model}</td>
        <td>${row.representation}</td>
        <td>${fmt(row.accuracy)}</td>
        <td>${fmt(row.macroF1)}</td>
      `;
      tbody.appendChild(tr);
    });

    const scaled = standardizeColumns(X);
    const logistic = trainSoftmax(scaled, y, Math.round(getVal("analysisSeed")) + 91);
    const logisticImp = Array.from({length: meta.p}, (_, j) =>
      mean(logistic.W.map(row => Math.abs(row[j])))
    );
    const tree = trainDecisionTree(scaled, y);
    const treeImp = normaliseImportances(tree.importances, meta.p);
    const order = meta.columns.map((name, i) => ({
      name,
      logistic: logisticImp[i],
      tree: treeImp[i]
    })).sort((a, b) => Math.max(b.logistic, b.tree) - Math.max(a.logistic, a.tree));

    const xaiBody = document.querySelector("#xaiTable tbody");
    xaiBody.innerHTML = "";
    order.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${row.name}</td>
        <td>${fmt(row.logistic)}</td>
        <td>${fmt(row.tree)}</td>
      `;
      xaiBody.appendChild(tr);
    });

    const rootFeature = tree.tree.feature !== null ? meta.columns[tree.tree.feature] : null;
    const rootThreshold = tree.tree.threshold;
    let xaiText = `A simple readable explanation layer is included instead of heavy SHAP-style tooling. Logistic regression importance is the average absolute coefficient size after standardising the original variables. Decision-tree importance is based on impurity reduction.`;
    if (rootFeature) {
      xaiText += ` The first tree split is on ${rootFeature} at about ${fmt(rootThreshold, 2)}, which makes it the most immediate rule-like separator in this small model.`;
    }
    xaiText += ` The strongest original-variable signals here are ${order.slice(0, 3).map(d => d.name).join(", ")}.`;
    document.getElementById("xaiSummary").textContent = xaiText;

    const repWins = {};
    results.forEach(row => {
      repWins[row.representation] = Math.max(repWins[row.representation] || 0, row.accuracy);
    });
    let explain = `This section compares KNN, logistic regression, LDA, and a shallow decision tree on the original variables, raw PCs, and rank PCs.`;
    explain += config.mode === "holdout"
      ? " Scores come from a reproducible train/test split."
      : " Scores come from reproducible out-of-fold predictions across the selected CV folds.";
    explain += ` ${meta.targetNote}`;
    explain += ` The best result here is ${best.model} on ${best.representation} with accuracy ${fmt(best.accuracy)}.`;
    explain += " SVM is left for later so the static app stays compact and easy to inspect.";
    document.getElementById("supervisedExplain").textContent = explain;
  }

  function updateFactorSection(meta, rawPca) {
    const pcaLoads = pcaLoadings(rawPca, 2);
    const fa = factorAnalysisApprox(meta.X, 2);
    for (let c = 0; c < 2; c++) {
      const pcaCol = pcaLoads.map(row => row[c]);
      const faCol = fa.loadings.map(row => row[c]);
      if (safeCorr(pcaCol, faCol) < 0) {
        fa.loadings.forEach(row => { row[c] *= -1; });
      }
    }

    drawLoadingMap("factorPcaMap", pcaLoads, meta.columns, "#2563eb");
    drawLoadingMap("factorFaMap", fa.loadings, meta.columns, "#0f766e");

    document.getElementById("factorCommunalityVal").textContent = fmt(mean(fa.communalities));
    document.getElementById("factorUniquenessVal").textContent = fmt(Math.max(...fa.uniqueness));
    document.getElementById("factorVariableVal").textContent = String(meta.p);
    document.getElementById("factorFocusVal").textContent = "total variance vs common variance";
    document.getElementById("factorPcaCaption").textContent =
      "PCA loadings come from the first two principal components and explain total variance.";
    document.getElementById("factorFaCaption").textContent =
      "Factor loadings come from a small principal-axis style approximation and emphasise common variance.";

    const tbody = document.querySelector("#factorTable tbody");
    tbody.innerHTML = "";
    meta.columns.forEach((name, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${name}</td>
        <td>${fmt(pcaLoads[i][0])}</td>
        <td>${fmt(pcaLoads[i][1])}</td>
        <td>${fmt(fa.loadings[i][0])}</td>
        <td>${fmt(fa.loadings[i][1])}</td>
        <td>${fmt(fa.uniqueness[i])}</td>
      `;
      tbody.appendChild(tr);
    });

    document.getElementById("factorExplain").textContent =
      `PCA and factor analysis often point in similar directions when variables move together, but factor analysis tries to separate shared structure from variable-specific noise. Here the mean communality is ${fmt(mean(fa.communalities))}, so on average that much variance is being treated as common-factor signal rather than uniqueness.`;
  }

  function correspondenceAnalysis(demo) {
    const total = sum(demo.counts.flat());
    const P = demo.counts.map(row => row.map(x => x / total));
    const rowMass = P.map(row => sum(row));
    const colMass = transpose(P).map(col => sum(col));
    const S = P.map((row, i) => row.map((x, j) =>
      (x - rowMass[i] * colMass[j]) / Math.sqrt(rowMass[i] * colMass[j])
    ));
    const A = S.map(rowA => S.map(rowB => dot(rowA, rowB)));
    const eig = topEigenPairs(A, 2);
    const eigenvalues = eig.values.map(v => Math.max(v, 0));
    const singular = eigenvalues.map(v => Math.sqrt(v));
    const totalInertia = sum(eigenvalues) || 1;
    const rowCoords = demo.rowLabels.map((_, i) => [0, 0]);
    const colCoords = demo.colLabels.map((_, j) => [0, 0]);

    for (let c = 0; c < 2; c++) {
      if (singular[c] < 1e-12) continue;
      const u = eig.vectors[c];
      const v = demo.colLabels.map((_, j) =>
        sum(S.map((row, i) => row[j] * u[i])) / singular[c]
      );
      for (let i = 0; i < rowCoords.length; i++) rowCoords[i][c] = u[i] * singular[c] / Math.sqrt(rowMass[i]);
      for (let j = 0; j < colCoords.length; j++) colCoords[j][c] = v[j] * singular[c] / Math.sqrt(colMass[j]);
    }

    return {
      rowCoords,
      colCoords,
      axisInertia: eigenvalues.map(v => v / totalInertia)
    };
  }

  function updateCaSection() {
    const ca = correspondenceAnalysis(CA_DEMO);
    const coords = ca.rowCoords.concat(ca.colCoords);
    const colors = CA_DEMO.rowLabels.map(() => "#2563eb").concat(CA_DEMO.colLabels.map(() => "#d97706"));
    const labels = CA_DEMO.rowLabels.concat(CA_DEMO.colLabels);
    const shapes = CA_DEMO.rowLabels.map(() => "circle").concat(CA_DEMO.colLabels.map(() => "square"));
    drawProjectionScatter("caMap", coords, colors, "CA axis 1", "CA axis 2", labels, shapes);

    document.getElementById("caNameVal").textContent = CA_DEMO.name;
    document.getElementById("caAxis1Val").textContent = fmt(ca.axisInertia[0]);
    document.getElementById("caAxis2Val").textContent = fmt(ca.axisInertia[1]);
    document.getElementById("caUseVal").textContent = "categorical counts, not continuous variables";
    document.getElementById("caCaption").textContent =
      "Rows and columns are plotted together. Nearby row/column points indicate associations above what independence would predict.";
    document.getElementById("caLegend").innerHTML =
      '<span><span class="swatch" style="background:#2563eb"></span>row categories</span>' +
      '<span><span class="swatch square" style="background:#d97706"></span>column categories</span>';
    document.getElementById("caExplain").textContent =
      "Use correspondence analysis when the data are counts in a two-way table, such as category by category frequencies. Use PCA when the data are continuous measured variables. CA rescales deviations from independence, so it answers a different question from PCA.";

    const thead = document.querySelector("#caTable thead");
    const tbody = document.querySelector("#caTable tbody");
    thead.innerHTML = `<tr><th>Row \\ Column</th>${CA_DEMO.colLabels.map(c => `<th>${c}</th>`).join("")}</tr>`;
    tbody.innerHTML = "";
    CA_DEMO.rowLabels.forEach((rowName, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${rowName}</td>${CA_DEMO.counts[i].map(v => `<td>${v}</td>`).join("")}`;
      tbody.appendChild(tr);
    });
  }

  function updateCore(meta, rawPca, rankPca, rankInput) {
    document.getElementById("activeTitle").textContent = meta.title + ": score agreement across methods";
    updateDatasetNote(meta);
    updateMetaPills(meta);
    updateLegend(meta);

    const corrPC1 = safeCorr(rawPca.scores[0], rankPca.scores[0]);
    const corrPC2 = safeCorr(rawPca.scores[1], rankPca.scores[1]);
    document.getElementById("corrPC1Val").textContent = fmt(corrPC1);
    document.getElementById("corrPC2Val").textContent = fmt(corrPC2);
    document.getElementById("rawVarVal").textContent = `${fmt(rawPca.variance[0])}, ${fmt(rawPca.variance[1])}`;
    document.getElementById("rankVarVal").textContent = `${fmt(rankPca.variance[0])}, ${fmt(rankPca.variance[1])}`;

    drawComparisonScatter(
      "scatterPC1",
      rawPca.scores[0],
      rankPca.scores[0],
      meta,
      "standardized PC1 score from raw-value PCA",
      "standardized PC1 score from rank PCA"
    );
    drawComparisonScatter(
      "scatterPC2",
      rawPca.scores[1],
      rankPca.scores[1],
      meta,
      "standardized PC2 score from raw-value PCA",
      "standardized PC2 score from rank PCA"
    );

    if (meta.mode === "simulated") {
      document.getElementById("captionPC1").textContent =
        "PC1 compares the same simulated observations scored in two ways. Differences mainly come from nonlinearity and outlier sensitivity.";
      document.getElementById("captionPC2").textContent =
        "PC2 often diverges more because secondary structure is where ranking and raw spacing usually start to disagree.";
    } else {
      document.getElementById("captionPC1").textContent =
        "If the points stay near the diagonal, raw-value PCA and rank PCA are telling a similar story on the real data.";
      document.getElementById("captionPC2").textContent =
        "PC2 is often the clearer place to see practical differences between raw spacing and rank ordering.";
    }

    drawBiplot("biplotRaw", rawPca.scores[0], rawPca.scores[1], rawPca.components[0], rawPca.components[1], meta, "#2563eb");
    drawBiplot("biplotRank", rankPca.scores[0], rankPca.scores[1], rankPca.components[0], rankPca.components[1], meta, "#7c3aed");
    document.getElementById("captionRawBiplot").textContent =
      "Points are observations in raw-PCA space. Arrows show variable loadings. Raw PCA reacts to actual magnitudes, so skew and extreme values can shape this geometry.";
    document.getElementById("captionRankBiplot").textContent =
      "Points are observations in rank-PCA space. Because this uses order information, extreme values matter less unless they strongly change ordering.";

    drawMatrixHeatmap("heatPearson", rawPca.cov, meta.columns, {mode: "diverging", min: -1, max: 1});
    drawMatrixHeatmap("heatSpearman", covarianceMatrix(standardizeColumns(rankInput)), meta.columns, {mode: "diverging", min: -1, max: 1});
    updateLoadingsTable(rawPca, rankPca, meta.columns);
  }

  function update() {
    const meta = getActiveDataset();
    updateControlVisibility(meta);
    updateExploreGuide(meta);
    const maxPcs = Math.min(meta.p, 5);
    const rawPca = pca(meta.X, maxPcs);
    const rankInput = rankColumns(meta.X);
    const rankPca = alignPca(rawPca, pca(rankInput, maxPcs));
    const rawStandard = rawPca.Xs;
    const rankStandard = standardizeColumns(rankInput);
    const heatIndices = sampleIndices(meta.n, Math.min(meta.n, Math.round(getVal("heatmapSize"))), Math.round(getVal("analysisSeed")) + 7, meta.target);
    const analysisIndices = sampleIndices(meta.n, Math.min(meta.n, Math.round(getVal("analysisSubset"))), Math.round(getVal("analysisSeed")) + 19, meta.target);

    updateCore(meta, rawPca, rankPca, rankInput);
    updateDistanceSection(meta, rawStandard, rankStandard, heatIndices);
    updateMdsSection(meta, rawPca, rankPca, rawStandard, rankStandard, analysisIndices);
    updateClusterSection(meta, rawPca, rankPca, rawStandard, rankStandard, analysisIndices);
    updateSupervisedSection(meta);
    updateFactorSection(meta, rawPca);
    updateCaSection();
  }

  update();
})();
