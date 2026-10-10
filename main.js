// FloodSight Q page. Renders precomputed results at once, then runs the live final sampling
// (one job on the backend chosen in the Select QPU dialog) through Python.
import { DATA } from './ui/data.js';
import { renderMap, renderPassList } from './ui/map.js';
import { renderGantt } from './ui/gantt.js';
import { renderDiversity, renderConnectivity } from './ui/charts.js';
import { table, fmt, pm, segmented, h } from './ui/util.js';
import * as api from 'fsq.api';

const SHOTS = 1024;
const METHODS = [
  ['exact', 'Exact optimum', 'var(--s1)'],
  ['greedy', 'Greedy', 'var(--s2)'],
  ['hybrid', 'Hybrid QAOA', 'var(--s3)'],
  ['sa', 'Simulated annealing', 'var(--s4)'],
];
const state = { inst: 'real', method: 'hybrid' };

function methodsFor(inst) {
  return METHODS.filter(([k]) => inst.schedules[k]).concat(inst.schedules.live ? [['live', 'Hybrid QAOA (this run)', 'var(--s3)']] : []);
}

function drawSchedules() {
  const inst = DATA.instances[state.inst];
  const ms = methodsFor(inst);
  if (!inst.schedules[state.method]) state.method = 'hybrid';
  segmented(document.getElementById('method-buttons'), ms.map(([k, l]) => [k, l]), state.method, (k) => { state.method = k; drawSchedules(); });
  const label = ms.find(([k]) => k === state.method)[1];
  const sel = inst.schedules[state.method];
  document.getElementById('map-caption').textContent =
    `${inst.label}. ${inst.candidates.length} possible images, ${inst.n_edges} conflicts. Showing: ${label}` +
    (inst.ratios[state.method] !== undefined ? ` (ratio to optimum ${fmt(inst.ratios[state.method])}).` : '.');
  renderMap(document.getElementById('map'), DATA, inst, sel);
  renderPassList(document.getElementById('pass-list'), DATA, inst, sel, label);
  const gs = ['exact', 'greedy', state.inst === 'default' && inst.schedules.live ? 'live' : 'hybrid']
    .map((k) => ({ key: k, label: ms.find(([m]) => m === k)?.[1] || k, color: METHODS.find(([m]) => m === k)?.[2] || 'var(--s3)', selected: inst.schedules[k] }))
    .filter((m) => m.selected);
  renderGantt(document.getElementById('gantt'), DATA, inst, gs);
}

function drawStatic() {
  const sel = document.getElementById('inst-select');
  sel.innerHTML = '';
  Object.entries(DATA.instances).forEach(([k, v]) => sel.appendChild(h('option', { value: k }, v.label)));
  sel.value = state.inst;
  sel.addEventListener('change', () => { state.inst = sel.value; drawSchedules(); });
  drawSchedules();

  const E = DATA.e1;
  document.getElementById('ratio-table').appendChild(table(
    ['Method', 'real event', 'mean ± sd (21 instances)', 'worst', 'runtime s (mean)'],
    E.rows.map((r) => [r.label, fmt(r.real), pm([r.mean, r.sd]), fmt(r.min), fmt(r.runtime, 2)]),
    E.note));

  const D = DATA.div;
  document.getElementById('div-caption').textContent = D.caption;
  renderDiversity(document.getElementById('div-chart'), document.getElementById('div-table'), D);

  let p = 1;
  const drawConn = () => renderConnectivity(document.getElementById('conn-chart'), document.getElementById('conn-table'), DATA.conn.rows, p);
  segmented(document.getElementById('conn-p'), [[1, 'QAOA p = 1'], [2, 'QAOA p = 2']], 1, (k) => { p = k; drawConn(); });
  document.getElementById('conn-caption').textContent = DATA.conn.caption;
  drawConn();

  const more = document.getElementById('more');
  DATA.more.forEach((block) => {
    more.appendChild(h('h3', {}, block.title));
    if (block.text) more.appendChild(h('p', { class: 'caption' }, block.text));
    more.appendChild(table(block.headers, block.rows));
  });
  document.getElementById('foot').textContent = DATA.meta.footer;
}

function drawLive(r) {
  const inst = DATA.instances.default;
  const order = [['exact', 'Exact optimum (precomputed MILP)'], ['greedy', 'Greedy (live)'], ['simulated', `Simulated annealing (${r.shots} restarts, live)`], ['hybrid', `Hybrid QAOA (${r.backend})`]];
  const rows = order.filter(([k]) => r.methods[k]).map(([k, l]) => {
    const m = r.methods[k];
    return [l, fmt(m.ratio), m.x.reduce((a, b) => a + b, 0), m.runtime_s === null ? '–' : fmt(m.runtime_s, 2)];
  });
  const box = document.getElementById('live-table');
  box.innerHTML = '';
  box.appendChild(table(['Method', 'ratio to optimum', 'images', 'time s'], rows,
    `Default instance (${inst.candidates.length} candidates). ${r.note}. Stitched schedule feasible before repair: ${r.raw_feasible}; repair actions: ${r.repair_log.length}.`));
  const sel = [];
  r.methods.hybrid.x.forEach((v, i) => { if (v) sel.push(i); });
  inst.schedules.live = sel;
  inst.ratios.live = r.methods.hybrid.ratio;
  state.inst = 'default'; state.method = 'live';
  document.getElementById('inst-select').value = 'default';
  drawSchedules();
}

drawStatic();
const status = document.getElementById('live-status');
status.textContent = `Running final QAOA sampling: all clusters in one job, ${SHOTS} shots. Precomputed results below are already loaded.`;
try {
  const out = await api.run_default.callPromising(backend, SHOTS);
  const r = JSON.parse(String(out));
  status.textContent = `Done in ${fmt(r.runtime_s, 1)} s on ${r.backend}. The map now shows this run's hybrid schedule.`;
  drawLive(r);
} catch (e) {
  status.textContent = `The live run failed: ${e && e.message ? e.message : e}. Precomputed results below are unaffected.`;
  console.log('live run error: ' + String(e));
}
