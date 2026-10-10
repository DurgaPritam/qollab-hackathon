// Small DOM / SVG helpers shared by the charts. No external libraries.

const NS = 'http://www.w3.org/2000/svg';

export function s(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

export function h(tag, attrs = {}, text = null) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  return e;
}

export function linear(d0, d1, r0, r1) {
  const f = (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  f.ticks = (n = 5) => {
    const step = niceStep((d1 - d0) / n);
    const out = [];
    for (let v = Math.ceil(d0 / step) * step; v <= d1 + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  };
  return f;
}

function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const m = raw / p;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10) * p;
}

export function fmt(v, d = 3) {
  return v === null || v === undefined || Number.isNaN(v) ? '–' : Number(v).toFixed(d);
}

export function pm(pair, d = 3) {
  return pair ? `${fmt(pair[0], d)} ± ${fmt(pair[1], d)}` : '–';
}

// A table from header + rows; returns a scroll wrapper.
export function table(headers, rows, caption = null) {
  const wrap = h('div', { class: 'tbl-wrap' });
  const t = h('table', { class: 't' });
  if (caption) t.appendChild(h('caption', { class: 'note', style: 'text-align:left' }, caption));
  const tr = h('tr');
  headers.forEach((x) => tr.appendChild(h('th', { scope: 'col' }, x)));
  const thead = h('thead'); thead.appendChild(tr); t.appendChild(thead);
  const tb = h('tbody');
  rows.forEach((r) => {
    const row = h('tr');
    r.forEach((c, i) => row.appendChild(h(i === 0 ? 'th' : 'td', i === 0 ? { scope: 'row' } : {}, String(c))));
    tb.appendChild(row);
  });
  t.appendChild(tb);
  wrap.appendChild(t);
  return wrap;
}

// One shared tooltip. Shown on hover and on focus/click so it works without a mouse.
let tipEl = null;
export function tip(target, html) {
  if (!tipEl) {
    tipEl = h('div', { class: 'tip', role: 'status' });
    document.body.appendChild(tipEl);
  }
  const show = (ev) => {
    tipEl.innerHTML = html;
    tipEl.style.display = 'block';
    const r = target.getBoundingClientRect();
    const x = ev && ev.clientX !== undefined ? ev.clientX : r.left + r.width / 2;
    const y = ev && ev.clientY !== undefined ? ev.clientY : r.top;
    tipEl.style.left = Math.min(x + 12, window.innerWidth - 270) + 'px';
    tipEl.style.top = y + 12 + 'px';
  };
  const hide = () => { tipEl.style.display = 'none'; };
  target.addEventListener('mousemove', show);
  target.addEventListener('mouseleave', hide);
  target.addEventListener('focus', show);
  target.addEventListener('blur', hide);
  target.setAttribute('tabindex', '0');
}

export function segmented(container, options, value, onChange) {
  container.innerHTML = '';
  options.forEach(([key, label]) => {
    const b = h('button', { type: 'button', role: 'radio', 'aria-checked': String(key === value) }, label);
    b.addEventListener('click', () => {
      container.querySelectorAll('button').forEach((x) => x.setAttribute('aria-checked', 'false'));
      b.setAttribute('aria-checked', 'true');
      onChange(key);
    });
    container.appendChild(b);
  });
}

export function legend(items) {
  const d = h('div', { class: 'legend' });
  items.forEach(([color, label]) => {
    const sp = h('span');
    const i = h('i'); i.style.background = color;
    sp.appendChild(i); sp.appendChild(document.createTextNode(label));
    d.appendChild(sp);
  });
  return d;
}
