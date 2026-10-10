// Results charts: diversity (grouped bars) and connectivity (lines). Each has a table view.
import { s, h, linear, tip, legend, table, fmt, pm } from './util.js';

// Grouped bars: one panel per metric, QAOA vs SA side by side, with ±1 sd whiskers.
export function renderDiversity(el, tblEl, div) {
  el.innerHTML = ''; tblEl.innerHTML = '';
  const metrics = [
    ['feasible_rate', 'Feasible rate', 1],
    ['n_good', 'Distinct near-optimal schedules (≥95%)', null],
    ['diversity', 'Mean Hamming distance among them', null],
  ];
  const series = [['qaoa', 'QAOA (warm start, p=1)', 'var(--s1)'], ['sa', 'Simulated annealing', 'var(--s2)']];
  el.appendChild(legend(series.map(([, l, c]) => [c, l])));
  const grid = h('div', { style: 'display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px' });
  el.appendChild(grid);
  metrics.forEach(([key, title, fixedMax]) => {
    const cell = h('div');
    cell.appendChild(h('div', { class: 'note', style: 'margin-bottom:4px' }, title));
    const W = 260, H = 170, l = 36, b = 22, t = 8;
    const vals = series.map(([k]) => div.cluster[k][key]);
    const ymax = (fixedMax || Math.max(...vals.map((v) => v[0] + v[1])) || 1) * 1.15;
    const y = linear(0, ymax, H - b, t);
    const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
      'aria-label': `${title}: ` + series.map(([k, l], i) => `${l} ${fmt(vals[i][0], 2)}`).join(', ') }, cell);
    y.ticks(4).filter((v) => !fixedMax || v <= fixedMax).forEach((v) => {
      s('line', { x1: l, x2: W - 6, y1: y(v), y2: y(v), stroke: 'var(--grid)' }, svg);
      const tx = s('text', { x: l - 4, y: y(v) + 4, 'text-anchor': 'end' }, svg); tx.textContent = v;
    });
    const bw = 56;
    series.forEach(([k, label, color], i) => {
      const [m, sd] = vals[i];
      const cx = l + (W - l) * (i === 0 ? 0.32 : 0.68);
      const bar = s('path', { d: roundedTop(cx - bw / 2, y(m), bw, y(0) - y(m), 4), fill: color }, svg);
      s('line', { x1: cx, x2: cx, y1: y(Math.min(ymax, m + sd)), y2: y(Math.max(0, m - sd)), stroke: 'var(--text-secondary)', 'stroke-width': 1.5 }, svg);
      const tx = s('text', { x: cx, y: y(Math.min(ymax, m + sd)) - 4, 'text-anchor': 'middle', class: 'lbl' }, svg);
      tx.textContent = fmt(m, 2);
      tip(bar, `<b>${label}</b><br>${title}: ${pm(vals[i], 3)} (mean ± sd over ${div.cluster.n} clusters)`);
    });
    s('line', { x1: l, x2: W - 6, y1: y(0), y2: y(0), stroke: 'var(--axis)' }, svg);
    grid.appendChild(cell);
  });
  const P = div.pipeline;
  tblEl.appendChild(table(
    ['Level / method', 'samples', 'feasible rate', 'distinct ≥95% schedules', 'diversity (Hamming)', 'best ratio'],
    [
      ...series.map(([k, l]) => [`Cluster · ${l}`, div.budget, pm(div.cluster[k].feasible_rate), pm(div.cluster[k].n_good, 1), pm(div.cluster[k].diversity, 2), pm(div.cluster[k].best_ratio)]),
      ['Full schedule · hybrid (stitched + repaired)', div.budget, pm(P.hybrid.feasible_rate), pm(P.hybrid.n_good, 1), pm(P.hybrid.diversity, 2), pm(P.hybrid.best_ratio)],
      ['Full schedule · SA raw', div.budget, pm(P.sa_raw.feasible_rate), pm(P.sa_raw.n_good, 1), pm(P.sa_raw.diversity, 2), pm(P.sa_raw.best_ratio)],
      ['Full schedule · SA + same repair', div.budget, pm(P.sa_repaired.feasible_rate), pm(P.sa_repaired.n_good, 1), pm(P.sa_repaired.diversity, 2), pm(P.sa_repaired.best_ratio)],
    ],
    `Mean ± sd. Cluster level: ${div.cluster.n} clusters with ≥4 candidates across ${div.n_instances} instances. Full schedule: ${div.n_instances} instances.`));
}

function roundedTop(x, y, w, hgt, r) {
  if (hgt <= 0) return `M${x},${y}h${w}`;
  r = Math.min(r, hgt, w / 2);
  return `M${x},${y + hgt}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + hgt}Z`;
}

// Lines: two-qubit gate count vs qubits for each connectivity.
export function renderConnectivity(el, tblEl, rows, p) {
  el.innerHTML = ''; tblEl.innerHTML = '';
  const data = rows.filter((r) => r.p === p);
  const series = [['all_to_all', 'All-to-all (trapped ion)', 'var(--s1)'], ['grid', 'Square grid', 'var(--s2)'], ['heavy_hex', 'Heavy-hex', 'var(--s3)']];
  el.appendChild(legend(series.map(([, l, c]) => [c, l])));
  const W = 760, H = 320, l = 56, r = 150, t = 12, b = 40;
  const xs = data.map((d) => d.n);
  const ymax = Math.max(...data.flatMap((d) => series.map(([k]) => d[k].two_qubit))) * 1.1;
  const x = linear(Math.min(...xs), Math.max(...xs), l, W - r);
  const y = linear(0, ymax, H - b, t);
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': 'Two-qubit gate count versus number of qubits for three connectivities. The table below has every value.' }, el);
  y.ticks(5).forEach((v) => {
    s('line', { x1: l, x2: W - r, y1: y(v), y2: y(v), stroke: 'var(--grid)' }, svg);
    const tx = s('text', { x: l - 6, y: y(v) + 4, 'text-anchor': 'end' }, svg); tx.textContent = v;
  });
  xs.forEach((v) => { const tx = s('text', { x: x(v), y: H - b + 16, 'text-anchor': 'middle' }, svg); tx.textContent = v; });
  const xl = s('text', { x: (l + W - r) / 2, y: H - 6, 'text-anchor': 'middle' }, svg); xl.textContent = 'qubits (candidates in the cluster)';
  const yl = s('text', { x: 14, y: (t + H - b) / 2, transform: `rotate(-90 14 ${(t + H - b) / 2})`, 'text-anchor': 'middle' }, svg); yl.textContent = 'two-qubit gates';
  s('line', { x1: l, x2: W - r, y1: y(0), y2: y(0), stroke: 'var(--axis)' }, svg);
  series.forEach(([k, label, color]) => {
    const d = data.map((row, i) => (i ? 'L' : 'M') + x(row.n).toFixed(1) + ',' + y(row[k].two_qubit).toFixed(1)).join('');
    s('path', { d, fill: 'none', stroke: color, 'stroke-width': 2 }, svg);
    data.forEach((row) => {
      const c = s('circle', { cx: x(row.n), cy: y(row[k].two_qubit), r: 4.5, fill: color, stroke: 'var(--surface-1)', 'stroke-width': 2 }, svg);
      tip(c, `<b>${label}</b>, ${row.n} qubits (density ${fmt(row.density, 2)})<br>2q gates ${row[k].two_qubit}, depth ${row[k].depth}<br>est. success ${fmt(row[k].est_success, 3)}`);
    });
  });
  // end labels, nudged apart so they never overlap
  const last = data[data.length - 1];
  const ends = series.map(([k, label]) => ({ label, y: y(last[k].two_qubit) + 4 })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.max(ends[i].y, ends[i - 1].y + 14);
  ends.forEach((e) => { const tx = s('text', { x: W - r + 8, y: e.y, class: 'lbl' }, svg); tx.textContent = e.label; });
  tblEl.appendChild(table(
    ['qubits', 'density', '2q all-to-all', '2q grid', '2q heavy-hex', 'depth all / grid / hex', 'est. success all / grid / hex'],
    data.map((d) => [d.n, fmt(d.density, 2), d.all_to_all.two_qubit, d.grid.two_qubit, d.heavy_hex.two_qubit,
      `${d.all_to_all.depth} / ${d.grid.depth} / ${d.heavy_hex.depth}`,
      `${fmt(d.all_to_all.est_success, 3)} / ${fmt(d.grid.est_success, 3)} / ${fmt(d.heavy_hex.est_success, 3)}`]),
    `QAOA p=${p}. Qiskit transpile, basis rz/sx/x/cx, optimization_level=3, seed_transpiler=42. est. success = (1 − 0.005)^(2q gates): a simple model, not a hardware measurement.`));
}
