// Region outline + targets, with the selected schedule highlighted. Plain SVG.
import { s, tip } from './util.js';

const PRIORITY_VARS = ['--p1', '--p2', '--p3', '--p4', '--p5'];

export function renderMap(el, data, inst, selected) {
  el.innerHTML = '';
  const W = 640, H = 420, pad = 28;
  const pts = [];
  data.region.forEach((ring) => ring.forEach((p) => pts.push(p)));
  data.targets.forEach((t) => pts.push([t.lon, t.lat]));
  const lons = pts.map((p) => p[0]), lats = pts.map((p) => p[1]);
  const lon0 = Math.min(...lons), lon1 = Math.max(...lons), lat0 = Math.min(...lats), lat1 = Math.max(...lats);
  const k = Math.cos(((lat0 + lat1) / 2) * Math.PI / 180);
  const sx = (W - 2 * pad) / ((lon1 - lon0) * k), sy = (H - 2 * pad) / (lat1 - lat0);
  const sc = Math.min(sx, sy);
  const ox = (W - (lon1 - lon0) * k * sc) / 2, oy = (H - (lat1 - lat0) * sc) / 2;
  const proj = ([lon, lat]) => [ox + (lon - lon0) * k * sc, oy + (lat1 - lat) * sc];

  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': `Map of ${data.targets.length} flood-affected locations. The list beside the map gives the same information as text.` }, el);
  data.region.forEach((ring) => {
    const d = ring.map((p, i) => (i ? 'L' : 'M') + proj(p).map((v) => v.toFixed(1)).join(',')).join('') + 'Z';
    s('path', { d, fill: 'var(--land)', stroke: 'var(--axis)', 'stroke-width': 1 }, svg);
  });

  // which targets are imaged in this schedule
  const imaged = new Map();
  selected.forEach((ci) => {
    const c = inst.candidates[ci];
    if (!imaged.has(c[0])) imaged.set(c[0], []);
    imaged.get(c[0]).push(c);
  });
  const inInstance = new Set(inst.candidates.map((c) => c[0]));
  const maxPop = Math.max(...data.targets.map((t) => t.population));
  const sorted = [...data.targets].sort((a, b) => b.population - a.population);
  sorted.forEach((t) => {
    const [x, y] = proj([t.lon, t.lat]);
    const r = 4 + 14 * Math.sqrt(t.population / maxPop);
    const g = s('g', {}, svg);
    const hit = imaged.has(t.target_id);
    const active = inInstance.has(t.target_id);
    s('circle', { cx: x, cy: y, r, fill: `var(${PRIORITY_VARS[t.priority - 1]})`,
      'fill-opacity': active ? 0.9 : 0.25, stroke: 'var(--surface-1)', 'stroke-width': 2 }, g);
    if (hit) s('circle', { cx: x, cy: y, r: r + 4, fill: 'none', stroke: 'var(--s2)', 'stroke-width': 2.5 }, g);
    const passes = imaged.get(t.target_id) || [];
    tip(g, `<b>${t.name}</b><br>priority ${t.priority} · ${t.population.toLocaleString()} ${data.meta.population_label}`
      + (passes.length ? `<br>imaged by ${passes.map((c) => `${c[1]} at +${c[2].toFixed(1)} h (${c[3].toFixed(0)}°)`).join(', ')}`
        : active ? '<br>not imaged in this schedule' : '<br>not in this instance'));
  });
  // labels largest-first; skip a label whose box would overlap one already placed
  const boxes = [];
  sorted.forEach((t) => {
    const [x, y] = proj([t.lon, t.lat]);
    const r = 4 + 14 * Math.sqrt(t.population / maxPop);
    const w = t.name.length * 6.2, hgt = 12;
    const tries = [[x + r + 3, y + 4, 'start'], [x - r - 3, y + 4, 'end'], [x, y - r - 4, 'middle'], [x, y + r + 12, 'middle']];
    for (const [lx, ly, anchor] of tries) {
      const x0 = anchor === 'start' ? lx : anchor === 'end' ? lx - w : lx - w / 2;
      const b = [x0, ly - hgt + 2, x0 + w, ly + 2];
      if (b[0] < 0 || b[2] > W) continue;
      if (boxes.some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1])) continue;
      boxes.push(b);
      const lb = s('text', { x: lx, y: ly, class: 'lbl', 'text-anchor': anchor }, svg);
      lb.textContent = t.name;
      break;
    }
  });
  // legend: priority ramp + ring
  const lg = s('g', { transform: `translate(${pad},${H - 14})` }, svg);
  const lt = s('text', { x: 0, y: 0 }, lg); lt.textContent = 'priority';
  for (let p = 1; p <= 5; p++) {
    s('circle', { cx: 52 + (p - 1) * 26, cy: -4, r: 6, fill: `var(${PRIORITY_VARS[p - 1]})` }, lg);
    const tx = s('text', { x: 52 + (p - 1) * 26 + 9, y: 0 }, lg); tx.textContent = p;
  }
  s('circle', { cx: 200, cy: -4, r: 6, fill: 'none', stroke: 'var(--s2)', 'stroke-width': 2.5 }, lg);
  const tr = s('text', { x: 211, y: 0 }, lg); tr.textContent = 'imaged in selected schedule';
}

export function renderPassList(el, data, inst, selected, label) {
  const names = Object.fromEntries(data.targets.map((t) => [t.target_id, t.name]));
  const rows = selected.map((ci) => inst.candidates[ci]).sort((a, b) => a[2] - b[2]);
  const total = selected.reduce((a, ci) => a + inst.candidates[ci][4], 0);
  el.innerHTML = `<p class="note"><b>${label}</b>: ${rows.length} images, value ${total.toFixed(3)}</p>`;
  const ol = document.createElement('ol');
  rows.forEach((c) => {
    const li = document.createElement('li');
    li.textContent = `+${c[2].toFixed(1)} h · ${names[c[0]] || c[0]} · ${c[1]} · ${c[3].toFixed(0)}° · ${c[5]}-looking`;
    ol.appendChild(li);
  });
  el.appendChild(ol);
}
