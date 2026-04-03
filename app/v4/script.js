(function() {
  const REAL_DATASETS = window.PCA_DATASETS || {};
  const ids = ["n","noise","nonlinear","outliers","outlierStrength","seed"];

  for (const id of ids) {
    const slider = document.getElementById(id);
    const num = document.getElementById(id + "_num");
    slider.addEventListener("input", () => { num.value = slider.value; update(); });
    num.addEventListener("input", () => { slider.value = num.value; update(); });
  }

  document.getElementById("datasetMode").addEventListener("change", update);
  document.getElementById("colorSpecial").addEventListener("change", update);
  document.getElementById("showLabels").addEventListener("change", update);
  document.getElementById("rerun").addEventListener("click", update);

  document.getElementById("presetLinear").addEventListener("click", () => {
    setVals({noise:0.25, nonlinear:0.10, outliers:2, outlierStrength:2.0});
  });
  document.getElementById("presetNonlinear").addEventListener("click", () => {
    setVals({noise:0.30, nonlinear:1.00, outliers:4, outlierStrength:3.0});
  });
  document.getElementById("presetOutliers").addEventListener("click", () => {
    setVals({noise:0.35, nonlinear:0.70, outliers:20, outlierStrength:9.0});
  });

  function setVals(obj) {
    for (const [k,v] of Object.entries(obj)) {
      document.getElementById(k).value = v;
      document.getElementById(k + "_num").value = v;
    }
    update();
  }

  function getVal(id) {
    return parseFloat(document.getElementById(id).value);
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
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function mean(arr) {
    return arr.reduce((a,b) => a+b, 0) / arr.length;
  }

  function std(arr) {
    const m = mean(arr);
    const v = arr.reduce((s,x) => s + (x-m)*(x-m), 0) / arr.length;
    return Math.sqrt(v);
  }

  function corr(a, b) {
    const ma = mean(a), mb = mean(b);
    let num = 0, va = 0, vb = 0;
    for (let i = 0; i < a.length; i++) {
      const da = a[i] - ma;
      const db = b[i] - mb;
      num += da * db;
      va += da * da;
      vb += db * db;
    }
    return num / Math.sqrt(va * vb);
  }

  function transpose(X) {
    return X[0].map((_, j) => X.map(row => row[j]));
  }

  function standardizeColumns(X) {
    const XT = transpose(X);
    const cols = XT.map(col => {
      const m = mean(col);
      const s = std(col) || 1;
      return col.map(x => (x - m) / s);
    });
    return transpose(cols);
  }

  function rankArray(arr) {
    const indexed = arr.map((v, i) => ({v, i})).sort((a,b) => a.v - b.v);
    const ranks = new Array(arr.length);
    let i = 0;
    while (i < indexed.length) {
      let j = i;
      while (j + 1 < indexed.length && indexed[j+1].v === indexed[i].v) j++;
      const avgRank = (i + j + 2) / 2;
      for (let k = i; k <= j; k++) ranks[indexed[k].i] = avgRank;
      i = j + 1;
    }
    return ranks;
  }

  function rankColumns(X) {
    const XT = transpose(X);
    return transpose(XT.map(rankArray));
  }

  function matVec(A, v) {
    return A.map(row => row.reduce((s, x, i) => s + x * v[i], 0));
  }

  function outer(v) {
    return v.map(vi => v.map(vj => vi * vj));
  }

  function subMatrices(A, B) {
    return A.map((row,i) => row.map((x,j) => x - B[i][j]));
  }

  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function norm(v) {
    return Math.sqrt(dot(v, v));
  }

  function scalarMult(v, c) {
    return v.map(x => x * c);
  }

  function covarianceMatrix(X) {
    const n = X.length;
    const p = X[0].length;
    const cov = Array.from({length: p}, () => Array(p).fill(0));
    for (let i = 0; i < p; i++) {
      for (let j = i; j < p; j++) {
        let s = 0;
        for (let r = 0; r < n; r++) s += X[r][i] * X[r][j];
        const val = s / (n - 1);
        cov[i][j] = val;
        cov[j][i] = val;
      }
    }
    return cov;
  }

  function powerIteration(A, maxIter=1500, tol=1e-11) {
    let v = Array(A.length).fill(0).map((_, i) => (i + 1) / A.length);
    let vnorm = norm(v) || 1;
    v = scalarMult(v, 1 / vnorm);
    for (let iter = 0; iter < maxIter; iter++) {
      let Av = matVec(A, v);
      const nrm = norm(Av) || 1;
      Av = scalarMult(Av, 1 / nrm);
      let diff = 0;
      for (let i = 0; i < v.length; i++) diff += Math.abs(Av[i] - v[i]);
      v = Av;
      if (diff < tol) break;
    }
    const lambda = dot(v, matVec(A, v));
    return {vector: v, value: lambda};
  }

  function pca2(X) {
    const Xs = standardizeColumns(X);
    const cov = covarianceMatrix(Xs);
    const eig1 = powerIteration(cov);
    const deflated = subMatrices(cov, outer(eig1.vector).map(row => row.map(x => x * eig1.value)));
    const eig2 = powerIteration(deflated);
    const scores1 = Xs.map(row => dot(row, eig1.vector));
    const scores2 = Xs.map(row => dot(row, eig2.vector));
    const totalVar = cov.reduce((s, row, i) => s + row[i], 0);
    return {
      Xs,
      cov,
      components: [eig1.vector.slice(), eig2.vector.slice()],
      eigenvalues: [eig1.value, eig2.value],
      scores: [scores1.slice(), scores2.slice()],
      variance: [eig1.value / totalVar, eig2.value / totalVar]
    };
  }

  function zscore(arr) {
    const m = mean(arr);
    const s = std(arr) || 1;
    return arr.map(x => (x - m) / s);
  }

  const palette = [
    "rgba(31,119,180,0.80)",
    "rgba(220,38,38,0.80)",
    "rgba(5,150,105,0.82)",
    "rgba(217,119,6,0.82)",
    "rgba(124,58,237,0.82)",
    "rgba(8,145,178,0.82)"
  ];

  function colorFor(index, meta) {
    const enabled = document.getElementById("colorSpecial").checked;
    if (!enabled) return "rgba(31,119,180,0.70)";
    if (meta.mode === "simulated") {
      return meta.specialFlags[index] ? "rgba(220,38,38,0.80)" : "rgba(31,119,180,0.70)";
    }
    if (meta.target && meta.target.length === meta.n) {
      return palette[meta.target[index] % palette.length];
    }
    return "rgba(31,119,180,0.70)";
  }

  function fmt(x, d=3) {
    return Number(x).toFixed(d);
  }

  function addLine(svg, x1, y1, x2, y2, stroke, width, opacity=1) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
    el.setAttribute("x1", x1); el.setAttribute("y1", y1);
    el.setAttribute("x2", x2); el.setAttribute("y2", y2);
    el.setAttribute("stroke", stroke); el.setAttribute("stroke-width", width);
    el.setAttribute("opacity", opacity);
    svg.appendChild(el);
  }

  function addCircle(svg, cx, cy, r, fill, stroke="none") {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    el.setAttribute("cx", cx); el.setAttribute("cy", cy);
    el.setAttribute("r", r);
    el.setAttribute("fill", fill);
    el.setAttribute("stroke", stroke);
    svg.appendChild(el);
  }

  function addText(svg, x, y, text, anchor, size, fill, weight="normal") {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
    el.setAttribute("x", x); el.setAttribute("y", y);
    el.setAttribute("text-anchor", anchor);
    el.setAttribute("font-size", size);
    el.setAttribute("fill", fill);
    el.setAttribute("font-weight", weight);
    el.textContent = text;
    svg.appendChild(el);
  }

  function addRect(svg, x, y, w, h, fill, stroke="none") {
    const el = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    el.setAttribute("x", x); el.setAttribute("y", y);
    el.setAttribute("width", w); el.setAttribute("height", h);
    el.setAttribute("fill", fill);
    el.setAttribute("stroke", stroke);
    svg.appendChild(el);
  }

  function fmtTick(x) {
    return Math.abs(x) >= 10 ? x.toFixed(0) : x.toFixed(1);
  }

  function alignComponents(raw, rank) {
    const out = JSON.parse(JSON.stringify(rank));
    for (let k = 0; k < 2; k++) {
      const c = corr(raw.scores[k], out.scores[k]);
      if (c < 0) {
        out.scores[k] = out.scores[k].map(v => -v);
        out.components[k] = out.components[k].map(v => -v);
      }
    }
    return out;
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
    const specialFlags = Array(n).fill(false);
    const cols = ["x1_linear", "x2_linear", "x3_exp", "x4_tanh", "x5_cubic"];

    for (let i = 0; i < n; i++) {
      const z = randn(rng);
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

    return {
      mode: "simulated",
      title: "Simulated dataset",
      subtitle: "Randomly generated with one hidden driver plus noise, monotone nonlinearity, and optional injected outliers.",
      X,
      columns: cols,
      specialFlags,
      target: null,
      target_names: null,
      n: X.length,
      p: cols.length
    };
  }

  function getRealDataset(key) {
    const ds = REAL_DATASETS[key];
    return {
      mode: "real",
      title: ds.name + " dataset",
      subtitle: "Built-in real dataset included directly inside the HTML.",
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

  function updateLegend(meta) {
    const legend = document.getElementById("legendRow");
    legend.innerHTML = "";
    const enabled = document.getElementById("colorSpecial").checked;
    if (!enabled) {
      legend.innerHTML = '<span><span class="swatch" style="background:rgba(31,119,180,0.70)"></span>points</span>';
      return;
    }
    if (meta.mode === "simulated") {
      legend.innerHTML =
        '<span><span class="swatch" style="background:rgba(31,119,180,0.70)"></span>regular points</span>' +
        '<span><span class="swatch" style="background:rgba(220,38,38,0.80)"></span>injected outliers</span>';
      return;
    }
    if (meta.target && meta.target_names) {
      let html = "";
      for (let i = 0; i < meta.target_names.length; i++) {
        html += `<span><span class="swatch" style="background:${palette[i % palette.length]}"></span>${meta.target_names[i]}</span>`;
      }
      legend.innerHTML = html;
      return;
    }
    legend.innerHTML = '<span><span class="swatch" style="background:rgba(31,119,180,0.70)"></span>points</span>';
  }

  function updateMetaPills(meta) {
    const el = document.getElementById("metaPills");
    let html = `<span class="pill">${meta.n} rows</span><span class="pill">${meta.p} variables</span>`;
    meta.columns.forEach(c => {
      html += `<span class="pill">${c}</span>`;
    });
    el.innerHTML = html;
  }

  function drawComparisonScatter(svgId, xVals, yVals, meta, xLabel, yLabel) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 700, H = 500;
    const m = {top: 24, right: 24, bottom: 58, left: 72};

    const zx = zscore(xVals);
    const zy = zscore(yVals);
    const all = zx.concat(zy);
    const minV = Math.min(...all), maxV = Math.max(...all);
    const pad = (maxV - minV) * 0.10 + 0.2;
    const xMin = minV - pad, xMax = maxV + pad;
    const yMin = minV - pad, yMax = maxV + pad;

    const sx = x => m.left + (x - xMin) / (xMax - xMin) * (W - m.left - m.right);
    const sy = y => H - m.bottom - (y - yMin) / (yMax - yMin) * (H - m.top - m.bottom);

    const ticks = 6;
    for (let i = 0; i <= ticks; i++) {
      const t = xMin + i * (xMax - xMin) / ticks;
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
    const ylabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    ylabel.setAttribute("x", 20);
    ylabel.setAttribute("y", H / 2);
    ylabel.setAttribute("transform", `rotate(-90 20 ${H/2})`);
    ylabel.setAttribute("text-anchor", "middle");
    ylabel.setAttribute("font-size", "15");
    ylabel.setAttribute("fill", "#111827");
    ylabel.textContent = yLabel;
    svg.appendChild(ylabel);
  }

  function drawBiplot(svgId, scores1, scores2, load1, load2, meta, titleColor) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 700, H = 520;
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
      addLine(svg, sx(0), sy(0), sx(x2), sy(y2), titleColor, 2.2);
      addCircle(svg, sx(x2), sy(y2), 4.5, titleColor);
      if (showLabels) {
        addText(svg, sx(x2) + 6, sy(y2) - 6, meta.columns[j], "start", 12, titleColor, "bold");
      }
    }

    addText(svg, W / 2, H - 14, "PC1 scores", "middle", 14, "#111827");
    const ylabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    ylabel.setAttribute("x", 18);
    ylabel.setAttribute("y", H / 2);
    ylabel.setAttribute("transform", `rotate(-90 18 ${H/2})`);
    ylabel.setAttribute("text-anchor", "middle");
    ylabel.setAttribute("font-size", "14");
    ylabel.setAttribute("fill", "#111827");
    ylabel.textContent = "PC2 scores";
    svg.appendChild(ylabel);
  }

  function colorMap(v) {
    const t = (v + 1) / 2;
    const r = Math.round(33 + t * 200);
    const b = Math.round(33 + (1 - t) * 200);
    const g = Math.round(50 + (1 - Math.abs(v)) * 120);
    return `rgb(${r},${g},${b})`;
  }

  function drawHeatmap(svgId, M, columns) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const W = 620, H = 560;
    const m = {top: 50, right: 80, bottom: 90, left: 110};
    const n = M.length;
    const cellW = (W - m.left - m.right) / n;
    const cellH = (H - m.top - m.bottom) / n;

    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const x = m.left + j * cellW;
        const y = m.top + i * cellH;
        addRect(svg, x, y, cellW, cellH, colorMap(M[i][j]), "#ffffff");
        addText(svg, x + cellW/2, y + cellH/2 + 4, fmt(M[i][j], 2), "middle", 12, "#111827");
      }
    }

    for (let i = 0; i < n; i++) {
      addText(svg, m.left + i * cellW + cellW/2, H - m.bottom + 22, columns[i], "middle", 11, "#111827");
      const lab = document.createElementNS("http://www.w3.org/2000/svg", "text");
      lab.setAttribute("x", 96);
      lab.setAttribute("y", m.top + i * cellH + cellH/2 + 4);
      lab.setAttribute("text-anchor", "end");
      lab.setAttribute("font-size", "11");
      lab.setAttribute("fill", "#111827");
      lab.textContent = columns[i];
      svg.appendChild(lab);
    }

    const lx = W - 48, ly = m.top, lh = n * cellH;
    for (let k = 0; k < 100; k++) {
      const v = 1 - 2 * (k / 99);
      addRect(svg, lx, ly + k * (lh/100), 18, lh/100 + 1, colorMap(v));
    }
    addText(svg, lx + 24, ly + 4, "1.0", "start", 11, "#111827");
    addText(svg, lx + 24, ly + lh/2 + 4, "0.0", "start", 11, "#111827");
    addText(svg, lx + 24, ly + lh, "-1.0", "start", 11, "#111827");
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

  function updateDatasetNote(meta) {
    const note = document.getElementById("datasetNote");
    if (meta.mode === "simulated") {
      note.textContent = meta.subtitle;
    } else {
      note.textContent = meta.subtitle + " Class coloring is shown when labels are available.";
    }
  }

  function update() {
    const meta = getActiveDataset();
    const raw = pca2(meta.X);
    const rankX = rankColumns(meta.X);
    let rank = pca2(rankX);
    rank = alignComponents(raw, rank);

    document.getElementById("activeTitle").textContent = meta.title + ": score agreement across methods";
    updateDatasetNote(meta);
    updateMetaPills(meta);
    updateLegend(meta);

    const corrPC1 = corr(raw.scores[0], rank.scores[0]);
    const corrPC2 = corr(raw.scores[1], rank.scores[1]);

    document.getElementById("corrPC1Val").textContent = fmt(corrPC1);
    document.getElementById("corrPC2Val").textContent = fmt(corrPC2);
    document.getElementById("rawVarVal").textContent = `${fmt(raw.variance[0])}, ${fmt(raw.variance[1])}`;
    document.getElementById("rankVarVal").textContent = `${fmt(rank.variance[0])}, ${fmt(rank.variance[1])}`;

    drawComparisonScatter(
      "scatterPC1",
      raw.scores[0],
      rank.scores[0],
      meta,
      "standardized PC1 score from raw-value PCA",
      "standardized PC1 score from rank PCA"
    );
    drawComparisonScatter(
      "scatterPC2",
      raw.scores[1],
      rank.scores[1],
      meta,
      "standardized PC2 score from raw-value PCA",
      "standardized PC2 score from rank PCA"
    );

    if (meta.mode === "simulated") {
      document.getElementById("captionPC1").textContent =
        "PC1 here compares the same random observations scored in two ways. Differences mainly come from nonlinearity and outlier sensitivity.";
      document.getElementById("captionPC2").textContent =
        "PC2 often differs more because it captures secondary structure after PC1 has already absorbed the dominant signal.";
    } else {
      document.getElementById("captionPC1").textContent =
        "PC1 here compares the same real observations scored in two ways. If the points stay near the diagonal, raw-value PCA and rank PCA are telling a similar story.";
      document.getElementById("captionPC2").textContent =
        "PC2 is a good place to look for practical differences between the two methods on real data, especially when scales, skew, or ordering effects matter.";
    }

    drawBiplot("biplotRaw", raw.scores[0], raw.scores[1], raw.components[0], raw.components[1], meta, "#2563eb");
    drawBiplot("biplotRank", rank.scores[0], rank.scores[1], rank.components[0], rank.components[1], meta, "#7c3aed");

    document.getElementById("captionRawBiplot").textContent =
      "Points are observations in raw-PCA space. Arrows show variable loadings. Raw PCA reacts to actual magnitudes, so skew and extreme values can shape this geometry.";
    document.getElementById("captionRankBiplot").textContent =
      "Points are observations in rank-PCA space. Because this uses order information, extreme values matter less unless they strongly change ordering.";

    drawHeatmap("heatPearson", raw.cov, meta.columns);
    drawHeatmap("heatSpearman", pca2(rankX).cov, meta.columns);

    updateLoadingsTable(raw, rank, meta.columns);
  }

  update();
})();
