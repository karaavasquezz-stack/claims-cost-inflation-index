// ── shared formatting helpers ────────────────────────────────────────────
const fmtEuro = v => v == null ? '—' : '€' + Math.round(v).toLocaleString('en-IE');
const fmtK    = v => {
  if (v == null) return '—';
  return Math.abs(v) >= 1000 ? '€' + (v/1000).toFixed(1) + 'k' : fmtEuro(v);
};
const fmtPct  = v => v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
const monthName = m => ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][m-1];

function setStatus(msg, cls) {
  const el = document.getElementById('status');
  if (!el) return;
  el.textContent = msg; el.className = cls || '';
}

function switchTab(name, names) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', names[i] === name));
  names.forEach(n => document.getElementById('tab-'+n).classList.toggle('active', n === name));
}

let charts = {};
function destroyChart(id) { if (charts[id]) { charts[id].destroy(); delete charts[id]; } }

// ── hover crosshair plugin: draws a dotted vertical line at the hovered x ──
const crosshairPlugin = {
  id: 'crosshair',
  afterDraw(chart) {
    const active = chart.getActiveElements();
    if (!active || !active.length) return;
    const { ctx, chartArea: { top, bottom } } = chart;
    const x = active[0].element.x;
    ctx.save();
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(26,26,26,0.35)';
    ctx.moveTo(x, top);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.restore();
  }
};
if (typeof Chart !== 'undefined') Chart.register(crosshairPlugin);

// shared interaction settings so the crosshair tracks smoothly on any chart
const CROSSHAIR_INTERACTION = { mode: 'index', intersect: false, axis: 'x' };

// ── Main forecast chart: actuals + Prophet + SARIMAX on one canvas ─────────
function buildDualForecastChart(canvasId, observed, prophetFc, sarimaxFc, yAxisMax) {
  destroyChart(canvasId);
  const obsLabels = observed.map(r => r.date.slice(0,7));
  const fcLabels  = prophetFc.map(r => r.date.slice(0,7));
  const allLabels = [...obsLabels, ...fcLabels];
  const pad = observed.length;

  const actuals = observed.map(r => r.avg_cost ? Math.round(r.avg_cost) : null);
  const smooth  = observed.map(r => r.avg_smooth ? Math.round(r.avg_smooth) : null);

  const pYhat = [...Array(pad).fill(null), ...prophetFc.map(r => Math.round(r.yhat))];
  const pHi   = [...Array(pad).fill(null), ...prophetFc.map(r => Math.round(r.yhat_upper))];
  const pLo   = [...Array(pad).fill(null), ...prophetFc.map(r => Math.round(r.yhat_lower))];
  const sYhat = [...Array(pad).fill(null), ...sarimaxFc.map(r => Math.round(r.yhat))];

  // Cap the y-axis using actuals + central forecast lines only — the 90% CI
  // band is allowed to clip off-chart at the top rather than stretching the
  // whole axis, since its extreme upper values aren't the headline number
  // anyone is reading off this chart.
  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    data: {
      labels: allLabels,
      datasets: [
        { type:'bar',  label:'Monthly actual', data:[...actuals,...Array(fcLabels.length).fill(null)],
          backgroundColor:'rgba(29,158,117,0.30)', borderColor:'#1D9E75', borderWidth:0.5 },
        { type:'line', label:'Smoothed', data:[...smooth,...Array(fcLabels.length).fill(null)],
          borderColor:'#085041', borderWidth:2, pointRadius:0, fill:false, spanGaps:true },
        { type:'line', label:'Prophet forecast', data:pYhat,
          borderColor:'#2563EB', borderWidth:2.5, pointRadius:0, fill:false, spanGaps:false },
        { type:'line', label:'Prophet 90% CI upper', data:pHi,
          borderColor:'transparent', borderWidth:0, pointRadius:0, fill:'+1', spanGaps:false,
          backgroundColor:'rgba(100,116,139,0.13)' },
        { type:'line', label:'Prophet 90% CI lower', data:pLo,
          borderColor:'transparent', borderWidth:0, pointRadius:0, fill:false, spanGaps:false },
        { type:'line', label:'SARIMAX forecast', data:sYhat,
          borderColor:'#D85A30', borderWidth:2.5, borderDash:[6,3], pointRadius:0, fill:false, spanGaps:false },
      ]
    },
    options: {
      responsive:true, maintainAspectRatio:false,
      interaction: CROSSHAIR_INTERACTION,
      plugins:{ legend:{display:false},
        tooltip:{ callbacks:{ label: ctx => ctx.dataset.label+': '+fmtK(ctx.raw) } } },
      scales:{
        x:{ ticks:{ maxTicksLimit:14, maxRotation:45 } },
        y:{ ticks:{ callback: v => fmtK(v) }, grid:{ color:'rgba(0,0,0,0.04)' }, max: yAxisMax || undefined }
      }
    }
  });
}

// ── YoY chart ────────────────────────────────────────────────────────────
function buildYoYChart(canvasId, observed, yMin, yMax) {
  destroyChart(canvasId);
  const rows = observed.filter(r => r.yoy != null);
  const ctx  = document.getElementById(canvasId).getContext('2d');
  const yScale = { ticks:{ callback: v => v+'%' } };
  if (yMin != null) yScale.min = yMin;
  if (yMax != null) yScale.max = yMax;
  charts[canvasId] = new Chart(ctx, {
    type:'bar',
    data:{
      labels: rows.map(r => r.date.slice(0,7)),
      datasets:[{ label:'YoY %', data: rows.map(r => parseFloat(r.yoy.toFixed(1))),
        backgroundColor: rows.map(r => r.yoy >= 0 ? 'rgba(29,158,117,0.55)' : 'rgba(216,90,48,0.55)'),
        borderWidth:0 }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction: CROSSHAIR_INTERACTION,
      plugins:{ legend:{display:false},
        tooltip:{ callbacks:{ label: ctx => 'YoY: '+fmtPct(ctx.raw) } } },
      scales:{ x:{ ticks:{ maxTicksLimit:10, maxRotation:45 } },
               y: yScale }
    }
  });
}

// ── Seasonality chart (from Prophet's monthly effect) ───────────────────
function buildSeasonalityChart(canvasId, seasonalityPct) {
  destroyChart(canvasId);
  const months = Array.from({length:12}, (_,i) => i+1);
  const vals = months.map(m => seasonalityPct[m] ?? seasonalityPct[String(m)] ?? 0);

  // Auto-scale to the data but round to a clean step so the axis doesn't
  // show cramped, jagged decimal ticks when the underlying effect is small.
  const maxAbs = Math.max(0.5, ...vals.map(v => Math.abs(v)));
  const niceMax = Math.ceil(maxAbs * 1.3 * 2) / 2; // round up to nearest 0.5

  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    type:'bar',
    data:{
      labels: months.map(monthName),
      datasets:[{ label:'Seasonal effect %', data: vals,
        backgroundColor: vals.map(v => v >= 0 ? 'rgba(29,158,117,0.6)' : 'rgba(216,90,48,0.6)'),
        borderRadius: 3,
        borderWidth:0 }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction: CROSSHAIR_INTERACTION,
      plugins:{
        legend:{display:false},
        tooltip:{ callbacks:{ label: ctx => fmtPct(ctx.raw) } },
      },
      scales:{
        x: { grid: { display:false } },
        y: {
          min: -niceMax, max: niceMax,
          ticks: { callback: v => v.toFixed(1)+'%', stepSize: niceMax/2 },
          grid: { color:'rgba(0,0,0,0.04)' },
        }
      }
    }
  });
}

// ── Dual seasonality chart (Prophet + SARIMAX side-by-side bars) ─────────
function buildDualSeasonalityChart(canvasId, prophetPct, sarimaxPct) {
  destroyChart(canvasId);
  const months = Array.from({length:12}, (_,i) => i+1);
  const pVals = months.map(m => prophetPct[m]  ?? prophetPct[String(m)]  ?? 0);
  const sVals = months.map(m => sarimaxPct[m]  ?? sarimaxPct[String(m)]  ?? null);
  const hasSarimax = sVals.some(v => v !== null);

  const allVals = [...pVals, ...(hasSarimax ? sVals.filter(v => v !== null) : [])];
  const maxAbs  = Math.max(0.5, ...allVals.map(v => Math.abs(v)));
  const niceMax = Math.ceil(maxAbs * 1.3 * 2) / 2;

  const datasets = [
    {
      label: 'Prophet',
      data: pVals,
      backgroundColor: pVals.map(v => v >= 0 ? 'rgba(37,99,235,0.65)' : 'rgba(37,99,235,0.35)'),
      borderRadius: 3, borderWidth: 0,
    }
  ];
  if (hasSarimax) {
    datasets.push({
      label: 'SARIMAX',
      data: sVals,
      backgroundColor: sVals.map(v => v == null ? 'transparent' : (v >= 0 ? 'rgba(216,90,48,0.65)' : 'rgba(216,90,48,0.35)')),
      borderRadius: 3, borderWidth: 0,
    });
  }

  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels: months.map(monthName), datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: CROSSHAIR_INTERACTION,
      plugins: {
        legend: { display: hasSarimax, position: 'top', labels: { boxWidth: 12, font: { size: 11 } } },
        tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + fmtPct(ctx.raw) } },
      },
      scales: {
        x: { grid: { display: false } },
        y: {
          min: -niceMax, max: niceMax,
          ticks: { callback: v => v.toFixed(1) + '%', stepSize: niceMax / 2 },
          grid: { color: 'rgba(0,0,0,0.04)' },
        }
      }
    }
  });
}

// ── Correlation bar chart (CPI vs HICP vs Construction, where present) ──
function buildCorrelationChart(canvasId, correlations) {
  destroyChart(canvasId);
  const labels = [];
  const vals = [];
  if (correlations.cpi_corr !== undefined) { labels.push('CPI'); vals.push(correlations.cpi_corr ?? 0); }
  if (correlations.hicp_corr !== undefined) { labels.push('HICP'); vals.push(correlations.hicp_corr ?? 0); }
  if (correlations.medical_corr !== undefined) { labels.push('Medical (CPI Health)'); vals.push(correlations.medical_corr ?? 0); }
  if (correlations.legal_corr !== undefined) { labels.push('Legal proxy (Prof. earnings)'); vals.push(correlations.legal_corr ?? 0); }
  if (correlations.construction_corr !== undefined) { labels.push('Construction wages'); vals.push(correlations.construction_corr ?? 0); }
  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    type:'bar',
    data:{
      labels,
      datasets:[{ label:'Pearson r', data: vals,
        backgroundColor: vals.map(v => v >= 0 ? 'rgba(37,99,235,0.6)' : 'rgba(216,90,48,0.6)'),
        borderWidth:0 }]
    },
    options:{
      indexAxis:'y',
      responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{display:false},
        tooltip:{ callbacks:{ label: ctx => 'r = '+ctx.raw.toFixed(3) } } },
      scales:{ x:{ min:-1, max:1 } }
    }
  });
}

// ── Backtest chart: actual vs Prophet vs SARIMAX over the held-out test year
function buildBacktestChart(canvasId, testPoints) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    data:{
      labels: testPoints.map(r => r.date.slice(0,7)),
      datasets:[
        { type:'line', label:'Actual', data: testPoints.map(r => r.actual),
          borderColor:'#1A1A1A', borderWidth:2.5, pointRadius:3, fill:false },
        { type:'line', label:'Prophet', data: testPoints.map(r => r.prophet),
          borderColor:'#2563EB', borderWidth:2, pointRadius:2, fill:false },
        { type:'line', label:'SARIMAX', data: testPoints.map(r => r.sarimax),
          borderColor:'#D85A30', borderWidth:2, borderDash:[5,3], pointRadius:2, fill:false },
      ]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction: CROSSHAIR_INTERACTION,
      plugins:{ legend:{display:true, position:'top', labels:{boxWidth:12,font:{size:11}}},
        tooltip:{ callbacks:{ label: ctx => ctx.dataset.label+': '+fmtK(ctx.raw) } } },
      scales:{ x:{ ticks:{ maxRotation:45 } }, y:{ ticks:{ callback: v => fmtK(v) } } }
    }
  });
}

// ── Severity breakdown chart (static — avg cost per tier, no forecast) ──
function buildSeverityChart(canvasId, rows) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  const labels = rows.map(r => r.tier);
  const avgCosts = rows.map(r => r.avg_cost);
  charts[canvasId] = new Chart(ctx, {
    type:'bar',
    data:{
      labels,
      datasets:[{ label:'Avg cost (€)', data: avgCosts,
        backgroundColor: 'rgba(29,158,117,0.6)',
        borderRadius: 4, borderWidth:0 }]
    },
    options:{
      indexAxis:'y',
      responsive:true, maintainAspectRatio:false,
      interaction: CROSSHAIR_INTERACTION,
      plugins:{ legend:{display:false},
        tooltip:{ callbacks:{ label: ctx => fmtEuro(ctx.raw) + ' avg' } } },
      scales:{ x:{ ticks:{ callback: v => fmtK(v) }, grid:{ color:'rgba(0,0,0,0.04)' } },
               y:{ grid:{ display:false } } }
    }
  });
}

function buildSeverityTable(tbodyId, rows) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = '';
  rows.forEach(r => {
    tbody.innerHTML += `<tr>
      <td>${r.tier}</td>
      <td class="num">${fmtEuro(r.avg_cost)}</td>
      <td class="num">${fmtEuro(r.median_cost)}</td>
      <td class="num">${r.n_claims.toLocaleString('en-IE')}</td>
      <td class="num">${r.pct_of_claims}%</td>
      <td class="num">${r.pct_of_cost}%</td>
    </tr>`;
  });
}

// ── Volume chart (claim counts per month) ───────────────────────────────
function buildVolChart(canvasId, observed) {
  destroyChart(canvasId);
  const ctx = document.getElementById(canvasId).getContext('2d');
  charts[canvasId] = new Chart(ctx, {
    type:'bar',
    data:{
      labels: observed.map(r => r.date.slice(0,7)),
      datasets:[{ label:'Claims', data: observed.map(r => r.n_claims),
        backgroundColor:'rgba(24,95,165,0.45)', borderWidth:0 }]
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction: CROSSHAIR_INTERACTION,
      plugins:{ legend:{display:false},
        tooltip:{ callbacks:{ label: ctx => 'Claims: '+ctx.raw } } },
      scales:{ x:{ ticks:{ maxTicksLimit:10, maxRotation:45 } }, y:{} }
    }
  });
}

// ── presets ──────────────────────────────────────────────────────────────
// Scenarios describe what type of macroeconomic environment to model.
// "bear" = low-inflation environment (e.g. ECB tightening, soft economy)
// "base" = central / most-likely outcome matching current CSO trend
// "bull" = high-pressure environment (persistent inflation, wage growth, legal reform delays)
//
// annual_cpi: annualised HICP projection rate fed directly to both models'
//   exogenous regressors (property page). Targets: bear ~2%, base ~4.5%, bull ~10%.
// sev: additional severity uplift % p.a. applied post-forecast via apply_scenario.
// legal: legal-cost multiplier in apply_scenario.
const PRESETS = {
  bear: { months:24, annual_cpi:2.0,  sev:0,   legal:1.0, covid:0, seas:0 },
  base: { months:24, annual_cpi:4.5,  sev:2.5, legal:1.1, covid:0, seas:0 },
  bull: { months:24, annual_cpi:10.0, sev:5.0, legal:1.3, covid:0, seas:0 },
};
function applyPreset(key, onApplied) {
  const p = PRESETS[key];
  const set = (id, val, fmt) => {
    const el = document.getElementById('sl-'+id);
    if (!el) return;
    el.value = val;
    document.getElementById('v-'+id).textContent = fmt(val);
  };
  set('months', p.months, v => v + ' mo');
  set('cpi',   p.cpi,   v => v.toFixed(1)+'%');
  set('sev',   p.sev,   v => v.toFixed(1)+'%');
  set('legal', p.legal, v => v.toFixed(2)+'×');
  set('covid', p.covid, v => (v>=0?'+':'')+v+'%');
  set('seas',  p.seas,  v => (v>=0?'+':'')+v+'%');
  if (onApplied) onApplied();
}

function buildTable(tbodyId, observed) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = '';
  [...observed].reverse().forEach(r => {
    const yoy = r.yoy != null
      ? `<span class="badge ${r.yoy>=0?'up':'dn'}">${fmtPct(r.yoy)}</span>`
      : '—';
    tbody.innerHTML += `<tr>
      <td>${r.date.slice(0,7)}</td>
      <td class="num">${fmtEuro(r.avg_cost)}</td>
      <td class="num">${r.n_claims}</td>
      <td class="num">${yoy}</td>
      <td class="num">${r.avg_smooth ? fmtEuro(r.avg_smooth) : '—'}</td>
    </tr>`;
  });
}

// ── Forecast numbers table (Prophet + SARIMAX side by side) ─────────────
function buildForecastTable(tbodyId, prophetFc, sarimaxFc) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody) return;
  tbody.innerHTML = '';
  const len = Math.max(prophetFc.length, sarimaxFc.length);
  for (let i = 0; i < len; i++) {
    const p = prophetFc[i];
    const s = sarimaxFc[i];
    const date = (p || s).date.slice(0, 7);
    tbody.innerHTML += `<tr>
      <td>${date}</td>
      <td class="num">${p ? fmtEuro(p.yhat) : '—'}</td>
      <td class="num" style="color:var(--muted);font-size:11px">${p ? fmtEuro(p.yhat_lower) + ' – ' + fmtEuro(p.yhat_upper) : '—'}</td>
      <td class="num">${s ? fmtEuro(s.yhat) : '—'}</td>
    </tr>`;
  }
}
