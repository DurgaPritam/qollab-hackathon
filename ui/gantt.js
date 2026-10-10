// Per-satellite timeline: grey ticks = every possible image, coloured marks = chosen images.
import { s, linear, tip, legend } from './util.js';

export function renderGantt(el, data, inst, schedules) {
  el.innerHTML = '';
  const names = Object.fromEntries(data.targets.map((t) => [t.target_id, t.name]));
  const sats = [...new Set(inst.candidates.map((c) => c[1]))].sort();
  const lanes = schedules.length;
  const laneH = 9, rowGap = 10, left = 110, right = 16, top = 8;
  const rowH = laneH * (lanes + 1) + rowGap;
  const W = 900, H = top + sats.length * rowH + 30;
  const x = linear(0, inst.horizon_h, left, W - right);
  el.appendChild(legend([['var(--muted)', 'possible image'], ...schedules.map((m) => [m.color, m.label])]));
  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': 'Timeline of imaging opportunities and chosen images per satellite. The pass list above lists the chosen images as text.' }, el);
  x.ticks(8).forEach((t) => {
    s('line', { x1: x(t), x2: x(t), y1: top, y2: H - 24, stroke: 'var(--grid)' }, svg);
    const tx = s('text', { x: x(t), y: H - 10, 'text-anchor': 'middle' }, svg); tx.textContent = `+${t} h`;
  });
  sats.forEach((sat, r) => {
    const y0 = top + r * rowH;
    const lb = s('text', { x: left - 8, y: y0 + rowH / 2, 'text-anchor': 'end', class: 'lbl' }, svg);
    lb.textContent = sat;
    s('line', { x1: left, x2: W - right, y1: y0 + rowH - rowGap / 2, y2: y0 + rowH - rowGap / 2, stroke: 'var(--grid)' }, svg);
    inst.candidates.forEach((c) => {
      if (c[1] !== sat) return;
      s('rect', { x: x(c[2]) - 1, y: y0, width: 2, height: laneH - 2, fill: 'var(--muted)', opacity: 0.6 }, svg);
    });
    schedules.forEach((m, li) => {
      m.selected.forEach((ci) => {
        const c = inst.candidates[ci];
        if (c[1] !== sat) return;
        const mark = s('rect', { x: x(c[2]) - 2.5, y: y0 + laneH * (li + 1), width: 5, height: laneH - 2, rx: 2, fill: m.color }, svg);
        tip(mark, `<b>${m.label}</b><br>${names[c[0]] || c[0]} by ${sat}<br>+${c[2].toFixed(2)} h, incidence ${c[3].toFixed(1)}°, value ${c[4].toFixed(3)}`);
      });
    });
  });
}
