/**
 * analysis.js
 * Pure-JS backend for PCA and Fisher Linear Discriminant Analysis (FLDA).
 * No external dependencies. All math operates on plain 2D arrays (row-major).
 *
 * Exported API (attach to window.Analysis for browser use):
 *   Analysis.parseCSV(text, labelCol)           → { X, labels, geneNames }
 *   Analysis.runPCA(X, nComponents)             → PCAResult
 *   Analysis.runFLDA(X, labels, classA, classB, nDirs) → FLDAResult
 */

(function (root) {
  "use strict";

  /* ─────────────────────────────────────────────
     Tiny matrix helpers
  ───────────────────────────────────────────── */

  /** Transpose a 2D array */
  function T(A) {
    const rows = A.length, cols = A[0].length;
    const B = Array.from({ length: cols }, () => new Float64Array(rows));
    for (let i = 0; i < rows; i++)
      for (let j = 0; j < cols; j++) B[j][i] = A[i][j];
    return B;
  }

  /** Matrix multiply A (m×k) · B (k×n) → m×n */
  function mmul(A, B) {
    const m = A.length, k = B.length, n = B[0].length;
    const C = Array.from({ length: m }, () => new Float64Array(n));
    for (let i = 0; i < m; i++)
      for (let p = 0; p < k; p++) {
        if (A[i][p] === 0) continue;
        for (let j = 0; j < n; j++) C[i][j] += A[i][p] * B[p][j];
      }
    return C;
  }

  /** Column-mean of matrix A (m×n) → Float64Array length n */
  function colMean(A) {
    const m = A.length, n = A[0].length;
    const mu = new Float64Array(n);
    for (let i = 0; i < m; i++)
      for (let j = 0; j < n; j++) mu[j] += A[i][j];
    for (let j = 0; j < n; j++) mu[j] /= m;
    return mu;
  }

  /** Centre matrix: subtract column means in place, return centred copy */
  function centre(A) {
    const mu = colMean(A);
    const B = A.map(row => {
      const r = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) r[j] = row[j] - mu[j];
      return r;
    });
    return { B, mu };
  }

  /** Dot product of two vectors */
  function dot(a, b) {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  /** L2 norm of a vector */
  function norm(v) { return Math.sqrt(dot(v, v)); }

  /** Normalize a vector in place */
  function normalise(v) {
    const n = norm(v);
    for (let i = 0; i < v.length; i++) v[i] /= n;
    return v;
  }

  /** Scale a vector by scalar */
  function scale(v, s) {
    const r = new Float64Array(v.length);
    for (let i = 0; i < v.length; i++) r[i] = v[i] * s;
    return r;
  }

  /** Add two vectors */
  function vadd(a, b) {
    const r = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) r[i] = a[i] + b[i];
    return r;
  }

  /** Subtract two vectors */
  function vsub(a, b) {
    const r = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) r[i] = a[i] - b[i];
    return r;
  }

  /* ─────────────────────────────────────────────
     SVD via randomised power iteration (thin SVD)
     Produces top-k left singular vectors (columns of U)
     and singular values (diagonal of S).
     A: m×n, k = number of components
  ───────────────────────────────────────────── */

  function randomisedSVD(A, k, nIter = 4, seed = 42) {
    const m = A.length, n = A[0].length;

    // Seeded pseudo-random for reproducibility
    let rng = seed;
    function rand() {
      rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
      return ((rng >>> 0) / 4294967296) * 2 - 1;
    }

    // Random test matrix Ω (n × k)
    const Omega = Array.from({ length: n }, () => {
      const row = new Float64Array(k);
      for (let j = 0; j < k; j++) row[j] = rand();
      return row;
    });

    // Y = A · Ω  (m × k)
    let Y = mmul(A, Omega);

    // Power iteration: Y = (A·A^T)^nIter · Y
    const At = T(A);
    for (let iter = 0; iter < nIter; iter++) {
      Y = mmul(A, mmul(At, Y));
    }

    // QR decomposition of Y (m × k) via Gram-Schmidt
    const Q = Array.from({ length: m }, () => new Float64Array(k));
    for (let j = 0; j < k; j++) {
      // Copy column j of Y into q
      let q = new Float64Array(m);
      for (let i = 0; i < m; i++) q[i] = Y[i][j];
      // Orthogonalise against previous columns
      for (let p = 0; p < j; p++) {
        let qp = new Float64Array(m);
        for (let i = 0; i < m; i++) qp[i] = Q[i][p];
        const proj = dot(q, qp);
        for (let i = 0; i < m; i++) q[i] -= proj * qp[i];
      }
      const nn = norm(q);
      if (nn < 1e-14) continue;
      for (let i = 0; i < m; i++) Q[i][j] = q[i] / nn;
    }

    // B = Q^T · A  (k × n)
    const Qt = T(Q);
    const B = mmul(Qt, A);

    // SVD of small B (k × n) via Jacobi for k ≤ ~20
    // We only need Vt and singular values
    const { U: Ub, S, Vt } = smallSVD(B, k);

    // U = Q · Ub  (m × k)
    const U = mmul(Q, Ub);

    return { U, S, Vt };
  }

  /* Small dense SVD (k×n, k small) using one-sided Jacobi */
  function smallSVD(A, k) {
    // Work on A·A^T (k×k) to find left singular vectors of A
    const At = T(A);
    const AAt = mmul(A, At); // k×k

    // Jacobi eigen-decomposition of symmetric AAt
    const { eigVecs, eigVals } = jacobiEigen(AAt, k);

    // Sort descending
    const order = eigVals.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    const sortedVals = order.map(o => Math.sqrt(Math.max(0, o[0])));
    const sortedVecs = order.map(o => eigVecs[o[1]]);

    // V = A^T · U / sigma
    const Vrows = [];
    for (let j = 0; j < k; j++) {
      if (sortedVals[j] < 1e-12) {
        Vrows.push(new Float64Array(A[0].length));
        continue;
      }
      // Compute A^T · u_j
      const u = sortedVecs[j];
      const v = new Float64Array(A[0].length);
      for (let n = 0; n < A[0].length; n++)
        for (let i = 0; i < k; i++) v[n] += At[n][i] * u[i];
      const vn = norm(v);
      if (vn > 1e-14) for (let n = 0; n < v.length; n++) v[n] /= vn;
      Vrows.push(v);
    }

    return {
      U: T(sortedVecs.map(v => Array.from(v))), // k×k transposed = k×k
      S: sortedVals,
      Vt: Vrows // k×n
    };
  }

  /* Jacobi eigen-decomp for small symmetric matrix A (n×n) */
  function jacobiEigen(A, n) {
    // Copy A
    const a = A.map(r => Array.from(r));
    // V = identity
    const V = Array.from({ length: n }, (_, i) => {
      const r = new Float64Array(n);
      r[i] = 1;
      return r;
    });

    const maxIter = 100 * n * n;
    for (let iter = 0; iter < maxIter; iter++) {
      // Find largest off-diagonal element
      let p = 0, q = 1, maxVal = Math.abs(a[0][1]);
      for (let i = 0; i < n - 1; i++)
        for (let j = i + 1; j < n; j++)
          if (Math.abs(a[i][j]) > maxVal) {
            maxVal = Math.abs(a[i][j]); p = i; q = j;
          }
      if (maxVal < 1e-12) break;

      // Compute Jacobi rotation angle
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
      const c = 1 / Math.sqrt(1 + t * t);
      const s = t * c;

      // Apply rotation to a
      const app = a[p][p], aqq = a[q][q], apq = a[p][q];
      a[p][p] = app - t * apq;
      a[q][q] = aqq + t * apq;
      a[p][q] = a[q][p] = 0;
      for (let r = 0; r < n; r++) {
        if (r === p || r === q) continue;
        const arp = a[r][p], arq = a[r][q];
        a[r][p] = a[p][r] = c * arp - s * arq;
        a[r][q] = a[q][r] = s * arp + c * arq;
      }
      // Apply rotation to eigenvectors V
      for (let r = 0; r < n; r++) {
        const vrp = V[r][p], vrq = V[r][q];
        V[r][p] = c * vrp - s * vrq;
        V[r][q] = s * vrp + c * vrq;
      }
    }

    return {
      eigVals: a.map((r, i) => r[i]),
      eigVecs: V        // columns are eigenvectors
    };
  }

  /* ─────────────────────────────────────────────
     CSV PARSING
  ───────────────────────────────────────────── */

  /**
   * Parse a CSV text. Labels are fully optional.
   * Auto-detects: sample ID column (first col, non-numeric header),
   * and optional label column by name match.
   *
   * @param {string} text - raw CSV
   * @param {string|null} labelCol - name of label column, or null/'' to skip
   * @returns {{ X: Float64Array[], labels: string[]|null, geneNames: string[], sampleIds: string[] }}
   *   labels is null if no label column found/requested.
   */
  function parseCSV(text, labelCol = null) {
    const lines = text.trim().split(/\r?\n/);
    if (lines.length < 2) throw new Error("CSV must have at least a header row and one data row.");
    const header = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));

    // Try to find label column (optional)
    let labelIdx = -1;
    if (labelCol && labelCol.trim()) {
      labelIdx = header.findIndex(h => h.toLowerCase() === labelCol.trim().toLowerCase());
    }

    // Auto-detect sample ID column: first column whose header is non-numeric
    // and not the label column, AND whose first data row value is also non-numeric
    let idIdx = -1;
    if (lines.length > 1) {
      const firstRow = splitCSVRow(lines[1]);
      const firstHeader = header[0];
      const firstVal = firstRow[0]?.trim() ?? "";
      const headerIsNonNumeric = firstHeader !== "" && isNaN(Number(firstHeader));
      const valueIsNonNumeric = isNaN(Number(firstVal));
      if ((headerIsNonNumeric || valueIsNonNumeric) && 0 !== labelIdx) {
        idIdx = 0;
      }
    }

    // All remaining columns are gene features
    const geneIndices = header
      .map((h, i) => ({ h, i }))
      .filter(({ i }) => i !== labelIdx && i !== idIdx)
      .map(({ i }) => i);

    if (geneIndices.length === 0) throw new Error("No numeric gene columns found after excluding ID/label columns.");
    const geneNames = geneIndices.map(i => header[i] || `gene_${i}`);

    const X = [], labels = [], sampleIds = [];

    for (let r = 1; r < lines.length; r++) {
      if (!lines[r].trim()) continue;
      const cols = splitCSVRow(lines[r]);
      if (labelIdx >= 0) {
        labels.push(cols[labelIdx]?.trim().replace(/^"|"$/g, "") ?? `unknown`);
      }
      sampleIds.push(idIdx >= 0 ? (cols[idIdx]?.trim() ?? `S${r}`) : `S${r}`);
      const row = new Float64Array(geneIndices.length);
      for (let j = 0; j < geneIndices.length; j++) {
        const v = parseFloat(cols[geneIndices[j]]);
        row[j] = isNaN(v) ? 0 : v;
      }
      X.push(row);
    }

    return {
      X,
      labels: labels.length > 0 ? labels : null,
      geneNames,
      sampleIds
    };
  }

  /* ─────────────────────────────────────────────
     K-MEANS CLUSTERING
  ───────────────────────────────────────────── */

  /**
   * K-Means++ clustering.
   * @param {Float64Array[]} X  - m × n matrix (operates on PC scores typically)
   * @param {number} k          - number of clusters
   * @param {number} maxIter    - max iterations (default 200)
   * @returns {{ labels: string[], centroids: Float64Array[], inertia: number, converged: boolean }}
   */
  function runKMeans(X, k, maxIter = 200) {
    const m = X.length;
    if (m === 0) throw new Error("Empty dataset for K-Means.");
    const n = X[0].length;
    k = Math.min(k, m);

    // K-Means++ initialisation
    let seed = 42;
    function rng() { seed ^= seed<<13; seed ^= seed>>17; seed ^= seed<<5; return (seed>>>0)/4294967296; }

    const centroidIdx = [Math.floor(rng() * m)];
    while (centroidIdx.length < k) {
      // Compute D² distances to nearest centroid
      const dists = X.map((x, i) => {
        let minD = Infinity;
        for (const ci of centroidIdx) {
          let d = 0;
          for (let j = 0; j < n; j++) { const dd = x[j] - X[ci][j]; d += dd*dd; }
          if (d < minD) minD = d;
        }
        return minD;
      });
      const totalD = dists.reduce((a, v) => a + v, 0);
      let r = rng() * totalD, cumD = 0;
      let chosen = m - 1;
      for (let i = 0; i < m; i++) { cumD += dists[i]; if (cumD >= r) { chosen = i; break; } }
      centroidIdx.push(chosen);
    }

    // Initialise centroids
    let centroids = centroidIdx.map(i => Float64Array.from(X[i]));
    let assignments = new Int32Array(m).fill(0);

    for (let iter = 0; iter < maxIter; iter++) {
      // Assignment step
      let changed = false;
      for (let i = 0; i < m; i++) {
        let bestK = 0, bestD = Infinity;
        for (let c = 0; c < k; c++) {
          let d = 0;
          for (let j = 0; j < n; j++) { const dd = X[i][j] - centroids[c][j]; d += dd*dd; }
          if (d < bestD) { bestD = d; bestK = c; }
        }
        if (assignments[i] !== bestK) { assignments[i] = bestK; changed = true; }
      }
      if (!changed) break;

      // Update step
      const newCentroids = Array.from({length: k}, () => new Float64Array(n));
      const counts = new Int32Array(k);
      for (let i = 0; i < m; i++) {
        const c = assignments[i];
        for (let j = 0; j < n; j++) newCentroids[c][j] += X[i][j];
        counts[c]++;
      }
      for (let c = 0; c < k; c++) {
        if (counts[c] > 0) {
          for (let j = 0; j < n; j++) newCentroids[c][j] /= counts[c];
          centroids[c] = newCentroids[c];
        }
      }
    }

    // Compute inertia
    let inertia = 0;
    for (let i = 0; i < m; i++) {
      const c = assignments[i];
      for (let j = 0; j < n; j++) { const dd = X[i][j] - centroids[c][j]; inertia += dd*dd; }
    }

    const labels = Array.from(assignments).map(c => `Cluster ${c + 1}`);
    return { labels, centroids, inertia, k };
  }

  /**
   * Run K-Means for k=2..maxK and return inertias (for elbow plot).
   */
  function elbowCurve(X, maxK = 10) {
    const results = [];
    for (let k = 1; k <= Math.min(maxK, X.length - 1); k++) {
      const r = runKMeans(X, k);
      results.push({ k, inertia: r.inertia });
    }
    return results;
  }

  /** Robust CSV row splitter (handles quoted commas) */
  function splitCSVRow(row) {
    const cols = [];
    let cur = "", inQ = false;
    for (let i = 0; i < row.length; i++) {
      const c = row[i];
      if (c === '"') { inQ = !inQ; }
      else if (c === ',' && !inQ) { cols.push(cur); cur = ""; }
      else cur += c;
    }
    cols.push(cur);
    return cols;
  }

  /* ─────────────────────────────────────────────
     PCA
  ───────────────────────────────────────────── */

  /**
   * Run PCA on data matrix X (m × n).
   * @param {Float64Array[]} X - m samples × n features
   * @param {number} nComponents - number of PCs to compute
   * @returns {PCAResult}
   *
   * PCAResult {
   *   scores:         Float64Array[]  m × nComponents   (projected samples)
   *   components:     Float64Array[]  nComponents × n   (loadings / eigenvectors)
   *   explainedVar:   Float64Array    variance per PC
   *   explainedVarRatio: Float64Array fraction per PC
   *   cumulativeVar:  Float64Array    cumulative fraction
   *   mean:           Float64Array    column means
   *   topGeneIndices: number[][]      top gene indices per PC
   * }
   */
  function runPCA(X, nComponents = 3) {
    const m = X.length;
    if (m === 0) throw new Error("Empty dataset");
    const n = X[0].length;
    const k = Math.min(nComponents, m, n);

    // Centre
    const { B, mu } = centre(X);

    // Randomised SVD of centred matrix (scaled by 1/sqrt(m-1) for covariance)
    const scale1 = 1 / Math.sqrt(m - 1);
    const Bsc = B.map(row => {
      const r = new Float64Array(n);
      for (let j = 0; j < n; j++) r[j] = row[j] * scale1;
      return r;
    });

    const { U, S, Vt } = randomisedSVD(Bsc, k);

    // Singular values → explained variance = S²  (eigenvalues of cov)
    const totalVar = S.reduce((a, v) => a + v * v, 0);

    // Also compute full total variance from trace of cov for ratio
    let traceCov = 0;
    for (let j = 0; j < n; j++) {
      let mu_j = 0;
      for (let i = 0; i < m; i++) mu_j += B[i][j];
      mu_j /= m;
      let v2 = 0;
      for (let i = 0; i < m; i++) { const d = B[i][j] - mu_j; v2 += d * d; }
      traceCov += v2 / (m - 1);
    }

    const explainedVar = new Float64Array(k);
    const explainedVarRatio = new Float64Array(k);
    const cumulativeVar = new Float64Array(k);
    let cumSum = 0;
    for (let i = 0; i < k; i++) {
      explainedVar[i] = S[i] * S[i];
      explainedVarRatio[i] = explainedVar[i] / traceCov;
      cumSum += explainedVarRatio[i];
      cumulativeVar[i] = cumSum;
    }

    // Scores = B · V^T^T = B · Vt^T   (m × k)
    const VtT = T(Vt);  // n × k
    const scores = mmul(B, VtT);

    // Top genes per PC: highest absolute loading
    const topGeneIndices = Vt.map(pc => {
      const abs = Array.from(pc).map((v, i) => [Math.abs(v), i]);
      return abs.sort((a, b) => b[0] - a[0]).map(x => x[1]);
    });

    return {
      scores,
      components: Vt,        // nComponents × n
      explainedVar,
      explainedVarRatio,
      cumulativeVar,
      mean: mu,
      topGeneIndices
    };
  }

  /* ─────────────────────────────────────────────
     FLDA (Fisher Linear Discriminant Analysis)
  ───────────────────────────────────────────── */

  /**
   * Run FLDA between two classes.
   * @param {Float64Array[]} X     - m × n feature matrix
   * @param {string[]} labels      - length m
   * @param {string} classA        - class label A
   * @param {string} classB        - class label B
   * @param {number} nDirs         - number of discriminant directions (1 or 2)
   * @returns {FLDAResult}
   *
   * FLDAResult {
   *   projection:  Float64Array[]  m×nDirs projected scores (only classA & classB samples)
   *   sampleIdx:   number[]        indices of included samples
   *   sampleLabels: string[]
   *   directions:  Float64Array[]  nDirs discriminant vectors (in PCA space if pcaResult given)
   *   trainAcc:    number
   *   testAcc:     number
   *   classA, classB
   * }
   */
  function runFLDA(X, labels, classA, classB, nDirs = 1, trainRatio = 0.8, pcaResult = null) {
    // Filter to the two classes
    const idxA = [], idxB = [];
    labels.forEach((l, i) => {
      if (l === classA) idxA.push(i);
      else if (l === classB) idxB.push(i);
    });
    if (idxA.length < 2 || idxB.length < 2)
      throw new Error(`Need ≥2 samples per class. Got ${idxA.length} for "${classA}", ${idxB.length} for "${classB}".`);

    // If PCA result provided, work in PC space
    let Xuse = X;
    if (pcaResult) Xuse = pcaResult.scores;

    const allIdx = [...idxA, ...idxB];
    const allLabels = allIdx.map(i => labels[i]);
    const XSub = allIdx.map(i => Xuse[i]);

    // Train/test split (deterministic shuffle via seeded index sort)
    const nA = idxA.length, nB = idxB.length;
    const trainIdxA = idxA.slice(0, Math.max(1, Math.round(nA * trainRatio)));
    const testIdxA  = idxA.slice(trainIdxA.length);
    const trainIdxB = idxB.slice(0, Math.max(1, Math.round(nB * trainRatio)));
    const testIdxB  = idxB.slice(trainIdxB.length);

    const trainIdx = [...trainIdxA, ...trainIdxB];
    const testIdx  = [...testIdxA, ...testIdxB];

    const XTrain = trainIdx.map(i => Xuse[i]);
    const yTrain = trainIdx.map(i => labels[i]);
    const XTest  = testIdx.map(i => Xuse[i]);
    const yTest  = testIdx.map(i => labels[i]);

    // Compute within-class scatter (Sw) and between-class scatter (Sb)
    const n = Xuse[0].length;

    const muA = colMean(trainIdxA.map(i => Xuse[i]));
    const muB = colMean(trainIdxB.map(i => Xuse[i]));
    const muAll = colMean(XTrain);

    // Sw
    const Sw = Array.from({ length: n }, () => new Float64Array(n));
    for (const cls of [trainIdxA, trainIdxB]) {
      const mu = cls === trainIdxA ? muA : muB;
      for (const i of cls) {
        const d = vsub(Xuse[i], mu);
        for (let r = 0; r < n; r++)
          for (let c = 0; c < n; c++) Sw[r][c] += d[r] * d[c];
      }
    }

    // Regularise Sw (add small ridge)
    const lambda = 1e-4;
    for (let i = 0; i < n; i++) Sw[i][i] += lambda;

    // Sw^-1 via Cholesky-like LU (or use power iteration on Sw^-1 · Sb)
    // Between-class scatter Sb = n_a(mu_a - mu)(mu_a-mu)^T + n_b(mu_b-mu)(mu_b-mu)^T
    const dA = vsub(muA, muAll);
    const dB = vsub(muB, muAll);

    const Sb = Array.from({ length: n }, () => new Float64Array(n));
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++)
        Sb[r][c] = trainIdxA.length * dA[r] * dA[c] + trainIdxB.length * dB[r] * dB[c];

    // Compute Sw^-1 · Sb via LU solve
    const SwInvSb = solveAB(Sw, Sb, n);

    // Get top eigenvectors of Sw^-1 · Sb
    const { eigVecs, eigVals } = jacobiEigen(SwInvSb, n);
    const order = eigVals.map((v, i) => [Math.abs(v), i]).sort((a, b) => b[0] - a[0]);
    const kDirs = Math.min(nDirs, n, 1); // max 1 direction for 2-class FLDA
    const directions = order.slice(0, kDirs).map(o => normalise(Float64Array.from(eigVecs[o[1]])));

    // Project all samples in XSub
    const projection = XSub.map(row => {
      const proj = new Float64Array(kDirs);
      for (let d = 0; d < kDirs; d++) proj[d] = dot(row, directions[d]);
      return proj;
    });

    // Compute threshold (midpoint of class means on LD1)
    const trainProjA = trainIdxA.map(i => dot(Xuse[i], directions[0]));
    const trainProjB = trainIdxB.map(i => dot(Xuse[i], directions[0]));
    const projMuA = trainProjA.reduce((a, v) => a + v, 0) / trainProjA.length;
    const projMuB = trainProjB.reduce((a, v) => a + v, 0) / trainProjB.length;
    const threshold = (projMuA + projMuB) / 2;

    // Classify: if projMuA > projMuB, above threshold = classA
    const predict = (score) => score >= threshold ? (projMuA >= projMuB ? classA : classB) : (projMuA >= projMuB ? classB : classA);

    // Train accuracy
    let correct = 0;
    for (let i = 0; i < XTrain.length; i++) {
      const score = dot(XTrain[i], directions[0]);
      if (predict(score) === yTrain[i]) correct++;
    }
    const trainAcc = correct / XTrain.length;

    // Test accuracy
    let testCorrect = 0;
    for (let i = 0; i < XTest.length; i++) {
      const score = dot(XTest[i], directions[0]);
      if (predict(score) === yTest[i]) testCorrect++;
    }
    const testAcc = XTest.length > 0 ? testCorrect / XTest.length : NaN;

    return {
      projection,
      sampleIdx: allIdx,
      sampleLabels: allLabels,
      directions,
      trainAcc,
      testAcc,
      threshold,
      projMuA,
      projMuB,
      classA,
      classB
    };
  }

  /* ─────────────────────────────────────────────
     LU solve:  solve A·X = B  (n×n systems)
  ───────────────────────────────────────────── */
  function solveAB(A, B, n) {
    // LU decomposition with partial pivoting
    const LU = A.map(r => Float64Array.from(r));
    const piv = Array.from({ length: n }, (_, i) => i);

    for (let k = 0; k < n; k++) {
      let maxVal = Math.abs(LU[k][k]), maxRow = k;
      for (let i = k + 1; i < n; i++)
        if (Math.abs(LU[i][k]) > maxVal) { maxVal = Math.abs(LU[i][k]); maxRow = i; }
      if (maxRow !== k) {
        [LU[k], LU[maxRow]] = [LU[maxRow], LU[k]];
        [piv[k], piv[maxRow]] = [piv[maxRow], piv[k]];
      }
      if (Math.abs(LU[k][k]) < 1e-14) continue;
      for (let i = k + 1; i < n; i++) {
        LU[i][k] /= LU[k][k];
        for (let j = k + 1; j < n; j++) LU[i][j] -= LU[i][k] * LU[k][j];
      }
    }

    // Solve for each column of B
    const X = Array.from({ length: n }, () => new Float64Array(n));
    for (let col = 0; col < n; col++) {
      // Permute B column
      const b = new Float64Array(n);
      for (let i = 0; i < n; i++) b[i] = B[piv[i]][col];
      // Forward substitution (L·y = b)
      for (let i = 0; i < n; i++)
        for (let j = 0; j < i; j++) b[i] -= LU[i][j] * b[j];
      // Back substitution (U·x = y)
      for (let i = n - 1; i >= 0; i--) {
        for (let j = i + 1; j < n; j++) b[i] -= LU[i][j] * b[j];
        if (Math.abs(LU[i][i]) > 1e-14) b[i] /= LU[i][i];
      }
      for (let i = 0; i < n; i++) X[i][col] = b[i];
    }
    return X;
  }

  /* ─────────────────────────────────────────────
     Demo data generator (TCGA-style, browser only)
  ───────────────────────────────────────────── */
  function generateDemoData(nSamples = 80, nGenes = 200) {
    const cancerTypes = ["Normal", "BRCA", "LUAD", "COAD"];
    const nPerClass = Math.floor(nSamples / cancerTypes.length);

    // Seeded RNG
    let seed = 12345;
    function rng() {
      seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
      return ((seed >>> 0) / 4294967296);
    }
    function randn() {
      const u = rng(), v = rng();
      return Math.sqrt(-2 * Math.log(u + 1e-10)) * Math.cos(2 * Math.PI * v);
    }

    // Each class has a distinct mean vector in gene space
    const classMeans = cancerTypes.map((_, c) => {
      const mu = new Float64Array(nGenes);
      for (let j = 0; j < nGenes; j++) mu[j] = (c === 0 ? 0 : (c === 1 ? 3 : (c === 2 ? -2 : 1.5))) * ((j % 20 < 10) ? 1.2 : 0.3) * randn();
      return mu;
    });

    const rows = [], labels = [], ids = [];
    cancerTypes.forEach((ct, c) => {
      for (let s = 0; s < nPerClass; s++) {
        const row = new Float64Array(nGenes);
        for (let j = 0; j < nGenes; j++)
          row[j] = classMeans[c][j] + randn() * 0.8;
        rows.push(row);
        labels.push(ct);
        ids.push(`${ct}_${s + 1}`);
      }
    });
    const geneNames = Array.from({ length: nGenes }, (_, i) => `GENE${i + 1}`);
    return { X: rows, labels, geneNames, sampleIds: ids };
  }

  /**
   * Parse a labels-only file.
   * Accepts: single-column CSV (with or without header), or one label per line plain text.
   * Auto-detects and skips a header row if the first row is non-numeric and only one unique
   * value exists in row 0 that doesn't appear in the rest (i.e. it looks like a column name).
   * @param {string} text
   * @returns {string[]} labels array
   */
  function parseLabelsFile(text) {
    const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
    if (!lines.length) throw new Error("Labels file is empty.");

    // Parse each line: take only first comma-separated token, strip quotes
    const tokens = lines.map(l => l.split(",")[0].trim().replace(/^"|"$/g, ""));

    // Heuristic: if first token looks like a header (all others ≥1 different value, and first token
    // never appears again), skip it
    const rest = tokens.slice(1);
    const firstIsHeader = rest.length > 0 && !rest.includes(tokens[0]) && rest.some(v => v !== tokens[0]);
    const labels = firstIsHeader ? rest : tokens;

    if (labels.length === 0) throw new Error("No labels found in file.");
    return labels;
  }

  /**
   * Load demo data from CSV files.
   * Intended for GitHub Pages/local-server use with:
   *   data/expression.csv
   *   data/labels.csv
   *
   * @param {string} expressionUrl - URL/path for expression matrix CSV
   * @param {string|null} labelsUrl - URL/path for labels file; pass null to skip
   * @returns {Promise<{ data: object, labels: string[]|null, expressionText: string, labelsText: string|null }>}
   */
  async function loadDemoData(expressionUrl = "data/expression.csv", labelsUrl = "data/labels.csv") {
    if (typeof fetch !== "function") {
      throw new Error("Demo CSV loading requires a browser or environment with fetch().");
    }

    const exprResponse = await fetch(expressionUrl);
    if (!exprResponse.ok) {
      throw new Error(`Could not load demo expression file: ${expressionUrl}`);
    }

    const expressionText = await exprResponse.text();
    const data = parseCSV(expressionText, null);

    let labels = null;
    let labelsText = null;
    if (labelsUrl) {
      const labelsResponse = await fetch(labelsUrl);
      if (!labelsResponse.ok) {
        throw new Error(`Could not load demo labels file: ${labelsUrl}`);
      }

      labelsText = await labelsResponse.text();
      labels = parseLabelsFile(labelsText);

      if (labels.length !== data.X.length) {
        throw new Error(
          `Demo label count mismatch: labels file has ${labels.length} rows but expression file has ${data.X.length} samples.`
        );
      }
    }

    return { data, labels, expressionText, labelsText };
  }

  /* Export */
  root.Analysis = {
    parseCSV,
    parseLabelsFile,
    runPCA,
    runKMeans,
    elbowCurve,
    runFLDA,
    generateDemoData,
    loadDemoData
  };

})(typeof window !== "undefined" ? window : global);
