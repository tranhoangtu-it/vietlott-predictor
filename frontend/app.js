/**
 * Vietlott AI Predictor - Full Frontend with Simulator
 * Static mode: loads pre-generated JSON data
 * LOAD ORDER: ai-autoplay-tab.js must load BEFORE this file (provides esc, loadAutoplay, etc.)
 */
const DATA_BASE = './data';
let currentGame = 'power655';
let currentPage = 1;
let totalPages = 1;
let predictionsData = null;
let analysisData = null;
let historyCache = {};
let gameConfig = null;

// Simulator state
let simSelected = [];
let simStats = { plays: 0, spent: 0, won: 0, bestMatch: 0, jackpots: 0 };

const TICKET_PRICE = {
    power655: 10000, mega645: 10000, keno: 10000,
    max3d: 10000, max3dplus: 10000,
    bingo18: 10000, power535: 10000,
};

const PRIZE_TABLE = {
    power655: { 6: 30_000_000_000, 5: 40_000_000, 4: 500_000, 3: 50_000 },
    mega645:  { 6: 12_000_000_000, 5: 10_000_000, 4: 300_000, 3: 30_000 },
    keno:     { 10: 2_000_000_000, 9: 500_000_000, 8: 50_000_000, 7: 5_000_000, 6: 500_000, 5: 50_000, 4: 10_000 },
    max3d:    { 3: 1_000_000_000, 2: 10_000_000, 1: 100_000 },
    max3dplus:{ 3: 1_000_000_000, 2: 10_000_000, 1: 100_000 },
    bingo18:  { 3: 2_000_000_000, 2: 500_000, 1: 20_000 },
    power535: { 6: 30_000_000_000, 5: 40_000_000, 4: 500_000, 3: 50_000 },
};

// ============================================
// Init
// ============================================
document.addEventListener('DOMContentLoaded', async () => {
    gameConfig = await fetchJSON(`${DATA_BASE}/game_config.json`);
    initNavigation();
    initGameSelect();
    initButtons();
    initSimulator();
    loadDashboard();
});

function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', e => {
            e.preventDefault();
            switchPage(link.dataset.page);
        });
    });
}

function switchPage(page) {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelector(`[data-page="${page}"]`).classList.add('active');
    document.getElementById(`page-${page}`).classList.add('active');
    ({ dashboard: loadDashboard, predictions: loadPredictions,
       history: loadHistory, analysis: loadAnalysis,
       simulator: setupSimulator, autoplay: loadAutoplay })[page]?.();
}

function initGameSelect() {
    document.getElementById('gameSelect').addEventListener('change', function() {
        currentGame = this.value;
        currentPage = 1;
        predictionsData = null;
        analysisData = null;
        simSelected = [];
        autoplayData = null;
        apCurrentPage = 1;
        const activePage = document.querySelector('.nav-link.active').dataset.page;
        switchPage(activePage);
    });
}

function initButtons() {
    const btnRefresh = document.getElementById('btnRefreshPred');
    const btnPrev = document.getElementById('btnPrevPage');
    const btnNext = document.getElementById('btnNextPage');
    if (btnRefresh) btnRefresh.addEventListener('click', () => { predictionsData = null; loadDashboard(); });
    if (btnPrev) btnPrev.addEventListener('click', () => { if (currentPage > 1) { currentPage--; loadHistory(); } });
    if (btnNext) btnNext.addEventListener('click', () => { if (currentPage < totalPages) { currentPage++; loadHistory(); } });
    initAutoplayPagination();
}

async function fetchJSON(url) {
    try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(r.status);
        return await r.json();
    } catch (e) {
        console.error(`Fetch: ${url}`, e);
        return null;
    }
}

function isDigitGame() {
    return ['max3d', 'max3dplus'].includes(currentGame);
}

function getGameInfo() {
    if (gameConfig && gameConfig[currentGame]) return gameConfig[currentGame];
    return { name: currentGame, max_number: 55, pick_count: 6, digit_game: false };
}

// ============================================
// Data Loading
// ============================================
async function loadPredictionsData() {
    if (predictionsData) return predictionsData;
    const all = await fetchJSON(`${DATA_BASE}/predictions.json`);
    if (all?.[currentGame]) predictionsData = all[currentGame];
    return predictionsData;
}

async function loadAnalysisData() {
    if (analysisData) return analysisData;
    analysisData = await fetchJSON(`${DATA_BASE}/${currentGame}_analysis.json`);
    return analysisData;
}

async function loadHistoryData() {
    if (historyCache[currentGame]) return historyCache[currentGame];
    const d = await fetchJSON(`${DATA_BASE}/${currentGame}_history.json`);
    if (d) historyCache[currentGame] = d;
    return d;
}

// ============================================
// Dashboard
// ============================================
async function loadDashboard() {
    const preds = await loadPredictionsData();
    if (preds) renderDashboard(preds);
    else document.getElementById('topPrediction').innerHTML = '<p style="text-align:center;color:var(--text-muted)">Chưa có dữ liệu.</p>';

    const analysis = await loadAnalysisData();
    if (analysis && !isDigitGame()) {
        renderHotCold(analysis);
        renderFrequencyChart(analysis);
    } else if (isDigitGame() && analysis) {
        renderDigitAnalysisDashboard(analysis);
    }
}

function renderDashboard(data) {
    document.getElementById('totalDraws').textContent = data.total_draws?.toLocaleString() || '--';
    if (gameConfig) document.getElementById('gameCount').textContent = Object.keys(gameConfig).length;
    // Show next draw info
    if (gameConfig?.[currentGame]?.schedule) {
        document.getElementById('nextDraw').textContent = calculateNextDraw(gameConfig[currentGame].schedule);
    }

    const allP = Object.values(data.predictions || {}).flat();
    if (allP.length) {
        const avg = allP.reduce((s, p) => s + (p.confidence || 0), 0) / allP.length;
        document.getElementById('avgConfidence').textContent = avg.toFixed(1) + '%';
    }
    // Show limited data warning banner
    const bannerEl = document.getElementById('topPrediction');
    if (data.limited_data) {
        bannerEl.insertAdjacentHTML('afterbegin',
            `<div style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);border-radius:8px;padding:12px 16px;margin-bottom:16px;color:#f59e0b;font-size:0.9rem">
                <i class="fas fa-exclamation-triangle"></i> Dữ liệu hạn chế (${data.total_draws} kỳ) — dự đoán mang tính ngẫu nhiên, chưa đủ dữ liệu để huấn luyện AI.
            </div>`);
    }
    renderTopPredictions(data.predictions);
    renderRecentDraws(data.recent_draws || []);
    if (data.analysis && !isDigitGame()) {
        analysisData = data.analysis;
        renderHotCold(data.analysis);
        renderFrequencyChart(data.analysis);
    }
}

function renderTopPredictions(predictions) {
    const c = document.getElementById('topPrediction');
    if (!predictions) { c.innerHTML = '<p style="text-align:center;color:var(--text-muted)">Chưa có</p>'; return; }

    const sources = isDigitGame()
        ? [{ key: 'digit_ai', label: 'Digit Pattern AI', icon: 'fas fa-hashtag' },
           { key: 'ensemble', label: 'Ensemble', icon: 'fas fa-layer-group' }]
        : [{ key: 'ensemble', label: 'Ensemble AI', icon: 'fas fa-layer-group' },
           { key: 'lstm', label: 'LSTM', icon: 'fas fa-brain' },
           { key: 'ml', label: 'RF + GB', icon: 'fas fa-tree' },
           { key: 'statistical', label: 'Statistical', icon: 'fas fa-chart-pie' }];

    let html = '';
    sources.forEach(s => {
        const preds = predictions[s.key];
        if (!preds?.length) return;
        const best = preds[0];
        const cc = best.confidence > 40 ? 'confidence-high' : best.confidence > 25 ? 'confidence-mid' : 'confidence-low';
        html += `<div class="prediction-set"><div class="pred-header">
            <span class="pred-method"><i class="${s.icon}"></i> ${s.label}</span>
            <span class="pred-confidence ${cc}">${best.confidence?.toFixed(1) || '?'}%</span>
            </div><div class="pred-numbers">
            ${best.numbers.map(n => `<span class="ball">${esc(n)}</span>`).join('')}
            </div></div>`;
    });
    c.innerHTML = html || '<p style="text-align:center;color:var(--text-muted)">Chưa có</p>';
}

function renderRecentDraws(draws) {
    const c = document.getElementById('recentDraws');
    if (!draws?.length) { c.innerHTML = '<p style="color:var(--text-muted)">Không có</p>'; return; }
    let html = '<div class="history-table"><table><thead><tr><th>#</th><th>Kết quả</th></tr></thead><tbody>';
    draws.slice(0, 8).reverse().forEach((d, i) => {
        html += `<tr><td style="color:var(--text-muted);font-family:'JetBrains Mono',monospace">${draws.length - i}</td><td>${d.map(n => `<span class="ball ball-sm">${esc(n)}</span>`).join(' ')}</td></tr>`;
    });
    c.innerHTML = html + '</tbody></table></div>';
}

function renderHotCold(a) {
    const c = document.getElementById('hotCold');
    if (!a) return;
    let html = '';
    if (a.hot_numbers) html += `<div class="hot-cold-section"><h3><i class="fas fa-fire" style="color:#ef4444"></i> Số nóng</h3><div class="balls-wrap">${a.hot_numbers.slice(0,10).map(n=>`<span class="ball ball-sm hot">${n}</span>`).join('')}</div></div>`;
    if (a.cold_numbers) html += `<div class="hot-cold-section"><h3><i class="fas fa-snowflake" style="color:#06b6d4"></i> Số lạnh</h3><div class="balls-wrap">${a.cold_numbers.slice(0,10).map(n=>`<span class="ball ball-sm cold">${n}</span>`).join('')}</div></div>`;
    if (a.overdue_numbers?.length) html += `<div class="hot-cold-section"><h3><i class="fas fa-clock" style="color:#8b5cf6"></i> Lâu chưa ra</h3><div class="balls-wrap">${a.overdue_numbers.slice(0,15).map(n=>`<span class="ball ball-sm overdue">${n}</span>`).join('')}</div></div>`;
    c.innerHTML = html;
}

function renderDigitAnalysisDashboard(a) {
    const hc = document.getElementById('hotCold');
    if (a.top_numbers) {
        hc.innerHTML = `<div class="hot-cold-section"><h3><i class="fas fa-fire" style="color:#ef4444"></i> Số xuất hiện nhiều nhất</h3><div class="balls-wrap">${a.top_numbers.slice(0,12).map(x=>`<span class="ball ball-sm hot">${x.number}</span>`).join('')}</div></div>`;
    }
    document.getElementById('frequencyChart').innerHTML = a.digit_frequency
        ? `<div style="padding:12px"><h4 style="margin-bottom:12px">Tần suất chữ số theo vị trí</h4>${Object.entries(a.digit_frequency).map(([pos,freq])=>`<div style="margin-bottom:16px"><strong>Vị trí ${parseInt(pos)+1}:</strong> ${Object.entries(freq).slice(0,10).map(([d,c])=>`<span class="ball ball-sm" style="width:32px;height:32px;font-size:0.75rem">${d}<small style="font-size:0.55rem;display:block">${c}</small></span>`).join('')}</div>`).join('')}</div>`
        : '';
}

function renderFrequencyChart(a) {
    const c = document.getElementById('frequencyChart');
    if (!a?.frequency) return;
    const freq = a.frequency;
    const nums = Object.keys(freq).map(Number).sort((a,b)=>a-b);
    const max = Math.max(...nums.map(n=>freq[n].count));

    // Show max 60 bars (for Keno 80 numbers it's still fine)
    let html = '<div class="freq-bar-container">';
    nums.forEach(n => {
        const cnt = freq[n].count;
        const pct = max > 0 ? cnt/max*100 : 0;
        const isHot = a.hot_numbers?.includes(n);
        const isCold = a.cold_numbers?.includes(n);
        const clr = isHot ? 'linear-gradient(90deg,#ef4444,#f97316)' : isCold ? 'linear-gradient(90deg,#06b6d4,#3b82f6)' : 'linear-gradient(90deg,#6366f1,#8b5cf6)';
        html += `<div class="freq-bar-row"><span class="freq-bar-label">${n}</span><div class="freq-bar-track"><div class="freq-bar-fill" style="width:${pct}%;background:${clr}"></div></div><span class="freq-bar-value">${cnt} (${freq[n].percentage}%)</span></div>`;
    });
    c.innerHTML = html + '</div>';
}

// ============================================
// Predictions Page
// ============================================
async function loadPredictions() {
    const data = await loadPredictionsData();
    if (!data) { document.getElementById('allPredictions').innerHTML = '<p style="text-align:center;color:var(--text-muted)">Chưa có.</p>'; return; }
    renderAllPredictions(data.predictions);
    renderMethodComparison();
}

function renderAllPredictions(preds) {
    const c = document.getElementById('allPredictions');
    if (!preds) return;
    const mcfg = {
        ensemble: { tag: 'tag-ensemble', label: 'Ensemble AI' },
        lstm: { tag: 'tag-lstm', label: 'LSTM Neural Network' },
        ml: { tag: 'tag-ml', label: 'Random Forest + GB' },
        statistical: { tag: 'tag-stat', label: 'Statistical' },
        digit_ai: { tag: 'tag-ensemble', label: 'Digit Pattern AI' },
    };
    let html = '';
    Object.entries(preds).forEach(([m, ps]) => {
        if (!ps?.length) return;
        const cfg = mcfg[m] || { tag: '', label: m };
        ps.forEach((p, i) => {
            const cc = p.confidence > 40 ? 'confidence-high' : p.confidence > 25 ? 'confidence-mid' : 'confidence-low';
            html += `<div class="prediction-card"><span class="method-tag ${cfg.tag}">${cfg.label} #${i+1}</span><div style="margin-bottom:8px"><span class="pred-confidence ${cc}" style="font-size:0.8rem">${p.confidence?.toFixed(1)||'?'}%</span></div><div class="pred-numbers">${p.numbers.map(n=>`<span class="ball">${esc(n)}</span>`).join('')}</div></div>`;
        });
    });
    c.innerHTML = html || '<p style="text-align:center;color:var(--text-muted)">Chưa có</p>';
}

function renderMethodComparison() {
    const c = document.getElementById('methodComparison');
    const methods = isDigitGame()
        ? [{ icon: 'fas fa-hashtag', name: 'Digit Pattern AI', desc: 'Phân tích tần suất chữ số theo từng vị trí, tìm pattern lặp lại.', color: '#ec4899' }]
        : [
            { icon: 'fas fa-layer-group', name: 'Ensemble AI', desc: 'Kết hợp LSTM (40%) + RF/GB (35%) + Thống kê (25%).', color: '#ec4899' },
            { icon: 'fas fa-brain', name: 'LSTM', desc: 'Mạng nơ-ron hồi quy phát hiện pattern chuỗi thời gian.', color: '#6366f1' },
            { icon: 'fas fa-tree', name: 'RF + Gradient Boosting', desc: 'Phân tích tần suất, khoảng cách, xu hướng chẵn/lẻ.', color: '#10b981' },
            { icon: 'fas fa-chart-pie', name: 'Statistical', desc: 'Số nóng/lạnh, cặp số phổ biến, phân bố tổng.', color: '#f59e0b' },
          ];
    c.innerHTML = methods.map(m => `<div class="method-card"><h3><i class="${m.icon}" style="color:${m.color}"></i> ${m.name}</h3><p>${m.desc}</p></div>`).join('');
}

// ============================================
// History
// ============================================
async function loadHistory() {
    const all = await loadHistoryData();
    if (!all?.length) { document.getElementById('historyTable').innerHTML = '<p style="text-align:center;color:var(--text-muted)">Chưa có.</p>'; return; }
    const pp = 20;
    totalPages = Math.ceil(all.length / pp);
    if (currentPage > totalPages) currentPage = totalPages;
    document.getElementById('pageInfo').textContent = `${currentPage}/${totalPages}`;
    renderHistory(all.slice((currentPage-1)*pp, currentPage*pp));
}

function renderHistory(draws) {
    const c = document.getElementById('historyTable');
    const hasPower = ['power655', 'power535'].includes(currentGame);
    const isBingo = currentGame === 'bingo18';
    const extraHeaders = hasPower ? '<th>Power</th>' : '';
    const bingoHeaders = isBingo ? '<th>Lớn/Nhỏ</th>' : '';
    let html = `<div class="history-table"><table><thead><tr><th>Kỳ</th><th>Ngày</th><th>Kết quả</th>${extraHeaders}${bingoHeaders}<th>Tổng</th></tr></thead><tbody>`;
    draws.forEach(d => {
        const nums = d.numbers || [];
        const sum = d.total || nums.reduce((s, n) => s + (typeof n === 'number' ? n : parseInt(n) || 0), 0);
        html += `<tr>
            <td style="font-weight:600;color:var(--primary-light);font-family:'JetBrains Mono',monospace">${esc(d.draw_id) || '--'}</td>
            <td style="color:var(--text-muted);font-size:0.9rem">${esc(d.date) || '--'}</td>
            <td>${nums.map(n => `<span class="ball ball-sm">${esc(n)}</span>`).join(' ')}</td>
            ${hasPower ? `<td><span class="ball ball-sm power">${esc(d.power) || '--'}</span></td>` : ''}
            ${isBingo ? `<td style="font-weight:500">${esc(d.large_small) || '--'}</td>` : ''}
            <td style="font-weight:600;font-family:'JetBrains Mono',monospace">${sum || '--'}</td>
        </tr>`;
    });
    c.innerHTML = html + '</tbody></table></div>';
}

// ============================================
// Analysis
// ============================================
async function loadAnalysis() {
    const d = await loadAnalysisData();
    if (!d) return;
    if (isDigitGame()) {
        renderDigitAnalysisPage(d);
    } else {
        renderOddEvenChart(d); renderDecadeChart(d); renderPairs(d); renderSumStats(d); renderOverdueNumbers(d);
    }
}

function renderDigitAnalysisPage(a) {
    document.getElementById('oddEvenChart').innerHTML = a.digit_frequency ? `<div style="padding:12px"><h4>Tần suất chữ số theo vị trí</h4>${Object.entries(a.digit_frequency).map(([p,f])=>`<div style="margin:12px 0"><strong>Vị trí ${parseInt(p)+1}:</strong><div class="balls-wrap" style="margin-top:6px">${Object.entries(f).slice(0,10).map(([d,c])=>`<span class="ball ball-sm">${d}<small style="font-size:0.6rem;display:block;opacity:0.7">${c}</small></span>`).join('')}</div></div>`).join('')}</div>` : '';
    document.getElementById('decadeChart').innerHTML = a.top_numbers ? `<div style="padding:12px"><h4>Top 20 số xuất hiện nhiều nhất</h4><div class="balls-wrap" style="margin-top:8px">${a.top_numbers.slice(0,20).map(x=>`<span class="ball ball-sm hot">${x.number} <small style="font-size:0.55rem;display:block">${x.count}x</small></span>`).join('')}</div></div>` : '';
    document.getElementById('pairsTable').innerHTML = '';
    document.getElementById('sumStats').innerHTML = a.digit_sum ? `<div class="sum-stat-item"><div class="value">${a.digit_sum.mean}</div><div class="label">TB tổng chữ số</div></div><div class="sum-stat-item"><div class="value">${a.digit_sum.std}</div><div class="label">Độ lệch chuẩn</div></div>` : '';
    document.getElementById('overdueNumbers').innerHTML = '';
}

function renderOddEvenChart(a) {
    const c = document.getElementById('oddEvenChart');
    if (!a.odd_even) return;
    let html = '<div style="display:flex;flex-direction:column;gap:8px;padding:10px">';
    Object.entries(a.odd_even).forEach(([o, pct]) => {
        html += `<div style="display:flex;align-items:center;gap:12px"><span style="width:80px;font-size:0.85rem;color:var(--text-muted)">${o}L/${6-parseInt(o)}C</span><div style="flex:1;height:28px;background:rgba(255,255,255,0.05);border-radius:6px;overflow:hidden"><div style="height:100%;width:${pct}%;background:var(--gradient-1);border-radius:6px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px"><span style="font-size:0.75rem;font-weight:600;color:white">${pct}%</span></div></div></div>`;
    });
    c.innerHTML = html + '</div>';
}

function renderDecadeChart(a) {
    const c = document.getElementById('decadeChart');
    if (!a.decade_distribution) return;
    const max = Math.max(...Object.values(a.decade_distribution));
    let html = '<div style="display:flex;flex-direction:column;gap:8px;padding:10px">';
    Object.entries(a.decade_distribution).forEach(([r, cnt]) => {
        const pct = max > 0 ? cnt/max*100 : 0;
        html += `<div style="display:flex;align-items:center;gap:12px"><span style="width:60px;font-size:0.85rem;color:var(--text-muted);text-align:right">${r}</span><div style="flex:1;height:28px;background:rgba(255,255,255,0.05);border-radius:6px;overflow:hidden"><div style="height:100%;width:${pct}%;background:var(--gradient-3);border-radius:6px;display:flex;align-items:center;padding-left:8px"><span style="font-size:0.75rem;font-weight:600;color:white">${cnt}</span></div></div></div>`;
    });
    c.innerHTML = html + '</div>';
}

function renderPairs(a) {
    const c = document.getElementById('pairsTable');
    if (!a.pairs) return;
    c.innerHTML = a.pairs.slice(0,15).map(p=>`<div class="pair-item"><div class="pair-numbers">${p.pair[0]} - ${p.pair[1]}</div><div class="pair-count">${p.count} lần</div></div>`).join('');
}

function renderSumStats(a) {
    const c = document.getElementById('sumStats');
    if (!a.sum_range) return;
    const s = a.sum_range;
    c.innerHTML = `<div class="sum-stat-item"><div class="value">${s.mean}</div><div class="label">Trung bình</div></div><div class="sum-stat-item"><div class="value">${s.median}</div><div class="label">Trung vị</div></div><div class="sum-stat-item"><div class="value">${s.std}</div><div class="label">Độ lệch chuẩn</div></div><div class="sum-stat-item"><div class="value">${s.min}</div><div class="label">Min</div></div><div class="sum-stat-item"><div class="value">${s.max}</div><div class="label">Max</div></div>`;
}

function renderOverdueNumbers(a) {
    const c = document.getElementById('overdueNumbers');
    if (!a.overdue_numbers) return;
    c.innerHTML = a.overdue_numbers.map(n=>`<span class="ball ball-sm overdue">${n}</span>`).join('');
}

// ============================================
// SIMULATOR
// ============================================
function initSimulator() {
    const bind = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); };
    bind('btnSimRandom', simRandom);
    bind('btnSimAI', simUseAI);
    bind('btnSimClear', simClear);
    bind('btnSimDraw', () => simDraw(1));
    bind('btnSimAuto', () => simDraw(100));
    bind('btnSimResetStats', simResetStats);
}

function setupSimulator() {
    simSelected = [];
    const info = getGameInfo();
    const grid = document.getElementById('simNumberGrid');
    const digitInput = document.getElementById('simDigitInput');
    const digitFields = document.getElementById('simDigitFields');

    if (isDigitGame()) {
        grid.style.display = 'none';
        digitInput.style.display = 'block';
        const digits = 3;
        const count = info.pick_count || 3;
        let html = '';
        for (let i = 0; i < count; i++) {
            html += `<input type="text" class="sim-digit-field" maxlength="${digits}" placeholder="${'0'.repeat(digits)}" data-idx="${i}" pattern="[0-9]*" inputmode="numeric">`;
        }
        digitFields.innerHTML = html;
    } else {
        grid.style.display = '';
        digitInput.style.display = 'none';
        const maxN = info.max_number || 55;
        const pickCount = currentGame === 'keno' ? 10 : (info.pick_count || 6);

        let html = '';
        for (let n = 1; n <= maxN; n++) {
            html += `<div class="sim-num" data-num="${n}">${n}</div>`;
        }
        grid.innerHTML = html;

        grid.querySelectorAll('.sim-num').forEach(el => {
            el.addEventListener('click', () => {
                const num = parseInt(el.dataset.num);
                if (simSelected.includes(num)) {
                    simSelected = simSelected.filter(x => x !== num);
                    el.classList.remove('selected');
                } else if (simSelected.length < pickCount) {
                    simSelected.push(num);
                    el.classList.add('selected');
                }
                updateSimSelectedDisplay();
            });
        });
    }
    updateSimSelectedDisplay();
    document.getElementById('simDrawResult').innerHTML = '<p style="color:var(--text-muted);text-align:center">Nhấn "Quay thưởng" để bắt đầu</p>';
    document.getElementById('simMatchResult').innerHTML = '';
    updateSimStats();
}

function updateSimSelectedDisplay() {
    const c = document.getElementById('simSelectedDisplay');
    if (isDigitGame()) {
        const fields = document.querySelectorAll('.sim-digit-field');
        const nums = [...fields].map(f => f.value).filter(v => v.length > 0);
        c.innerHTML = nums.length ? `<div class="pred-numbers">${nums.map(n=>`<span class="ball">${n}</span>`).join('')}</div>` : '<p style="color:var(--text-muted);font-size:0.9rem">Nhập số bên dưới</p>';
    } else {
        c.innerHTML = simSelected.length ? `<div class="pred-numbers">${simSelected.sort((a,b)=>a-b).map(n=>`<span class="ball">${n}</span>`).join('')}</div>` : '<p style="color:var(--text-muted);font-size:0.9rem">Chọn số bên dưới hoặc nhấn Random/AI</p>';
    }
}

function simRandom() {
    const info = getGameInfo();
    if (isDigitGame()) {
        const digits = 3;
        const count = info.pick_count || 3;
        const fields = document.querySelectorAll('.sim-digit-field');
        fields.forEach(f => {
            const max = Math.pow(10, digits);
            f.value = String(Math.floor(Math.random() * max)).padStart(digits, '0');
        });
    } else {
        const maxN = info.max_number || 55;
        const pick = currentGame === 'keno' ? 10 : (info.pick_count || 6);
        simSelected = [];
        const pool = Array.from({ length: maxN }, (_, i) => i + 1);
        for (let i = pool.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [pool[i], pool[j]] = [pool[j], pool[i]];
        }
        simSelected = pool.slice(0, pick).sort((a, b) => a - b);

        document.querySelectorAll('.sim-num').forEach(el => {
            el.classList.toggle('selected', simSelected.includes(parseInt(el.dataset.num)));
        });
    }
    updateSimSelectedDisplay();
}

async function simUseAI() {
    const preds = await loadPredictionsData();
    if (!preds?.predictions) { showToast('Chưa có dữ liệu AI', 'error'); return; }

    const firstMethod = Object.values(preds.predictions).find(arr => arr?.length);
    if (!firstMethod?.length) return;
    const aiNums = firstMethod[0].numbers;

    if (isDigitGame()) {
        const fields = document.querySelectorAll('.sim-digit-field');
        aiNums.forEach((n, i) => { if (fields[i]) fields[i].value = String(n); });
    } else {
        simSelected = aiNums.map(Number).filter(n => !isNaN(n));
        document.querySelectorAll('.sim-num').forEach(el => {
            el.classList.toggle('selected', simSelected.includes(parseInt(el.dataset.num)));
        });
    }
    updateSimSelectedDisplay();
    showToast('Đã dùng dự đoán AI!', 'success');
}

function simClear() {
    simSelected = [];
    document.querySelectorAll('.sim-num').forEach(el => el.classList.remove('selected'));
    document.querySelectorAll('.sim-digit-field').forEach(f => f.value = '');
    updateSimSelectedDisplay();
}

function simDraw(rounds) {
    let myNums;
    if (isDigitGame()) {
        const fields = document.querySelectorAll('.sim-digit-field');
        myNums = [...fields].map(f => f.value).filter(v => v.length > 0);
        if (myNums.length === 0) { showToast('Hãy nhập số trước!', 'error'); return; }
    } else {
        if (simSelected.length === 0) { showToast('Hãy chọn số trước!', 'error'); return; }
        myNums = [...simSelected];
    }

    const info = getGameInfo();
    const resultContainer = document.getElementById('simDrawResult');
    const matchContainer = document.getElementById('simMatchResult');

    let lastDraw, lastMatch = 0, lastPrize = 0;

    for (let r = 0; r < rounds; r++) {
        let drawnNums;
        if (isDigitGame()) {
            const digits = 3;
            const count = info.pick_count || 3;
            drawnNums = [];
            for (let i = 0; i < count; i++) {
                drawnNums.push(String(Math.floor(Math.random() * Math.pow(10, digits))).padStart(digits, '0'));
            }
            // Count exact matches
            lastMatch = myNums.filter(n => drawnNums.includes(n)).length;
        } else if (currentGame === 'keno') {
            // Keno: house draws 20 from 80
            const pool = Array.from({ length: 80 }, (_, i) => i + 1);
            for (let i = pool.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [pool[i], pool[j]] = [pool[j], pool[i]];
            }
            drawnNums = pool.slice(0, 20).sort((a, b) => a - b);
            lastMatch = myNums.filter(n => drawnNums.includes(n)).length;
        } else {
            // Standard number games (Power 6/55, Mega 6/45, Bingo 18, Power 5/35)
            const maxN = info.max_number || 55;
            const drawCount = info.pick_count || 6;
            const pool = Array.from({ length: maxN }, (_, i) => i + 1);
            for (let i = pool.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [pool[i], pool[j]] = [pool[j], pool[i]];
            }
            drawnNums = pool.slice(0, drawCount).sort((a, b) => a - b);
            lastMatch = myNums.filter(n => drawnNums.includes(n)).length;
        }

        lastDraw = drawnNums;

        // Calculate prize
        const prizes = PRIZE_TABLE[currentGame] || {};
        lastPrize = prizes[lastMatch] || 0;

        simStats.plays++;
        simStats.spent += TICKET_PRICE[currentGame] || 10000;
        simStats.won += lastPrize;
        if (lastMatch > simStats.bestMatch) simStats.bestMatch = lastMatch;
        if (lastPrize >= 1_000_000_000) simStats.jackpots++;
    }

    // Show last draw result
    resultContainer.innerHTML = `
        <div style="text-align:center;margin-bottom:16px">
            <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:12px">Kết quả quay${rounds > 1 ? ` (lần cuối trong ${rounds} lượt)` : ''}:</p>
            <div class="pred-numbers" style="justify-content:center">
                ${lastDraw.map(n => {
                    const isMatch = myNums.includes(isDigitGame() ? n : parseInt(n)) || myNums.includes(String(n));
                    return `<span class="ball ${isMatch ? 'hot' : ''}">${n}</span>`;
                }).join('')}
            </div>
        </div>
    `;

    // Show match result
    const pick = isDigitGame() ? myNums.length : (currentGame === 'keno' ? 10 : (info.pick_count || 6));
    const matchPct = pick > 0 ? (lastMatch / pick * 100).toFixed(0) : 0;
    const prizeFormatted = lastPrize > 0 ? lastPrize.toLocaleString('vi-VN') + ' VNĐ' : 'Không trúng';
    const emoji = lastMatch >= pick ? '🎉🎉🎉 JACKPOT!' : lastMatch >= pick - 1 ? '🎉 Gần trúng!' : lastMatch >= 3 ? '👍 Có thưởng!' : '😢';

    matchContainer.innerHTML = `
        <div class="sim-match-box ${lastPrize > 0 ? 'win' : 'lose'}">
            <div class="sim-match-emoji">${emoji}</div>
            <div class="sim-match-text">Trùng <strong>${lastMatch}/${pick}</strong> số (${matchPct}%)</div>
            <div class="sim-match-prize">${prizeFormatted}</div>
        </div>
    `;

    updateSimStats();
}

function updateSimStats() {
    document.getElementById('simTotalPlays').textContent = simStats.plays.toLocaleString();
    document.getElementById('simTotalSpent').textContent = simStats.spent.toLocaleString('vi-VN');
    document.getElementById('simTotalWon').textContent = simStats.won.toLocaleString('vi-VN');
    const profit = simStats.won - simStats.spent;
    const profitEl = document.getElementById('simProfit');
    profitEl.textContent = (profit >= 0 ? '+' : '') + profit.toLocaleString('vi-VN');
    profitEl.style.color = profit >= 0 ? '#10b981' : '#ef4444';
    document.getElementById('simBestMatch').textContent = simStats.bestMatch;
    document.getElementById('simJackpotCount').textContent = simStats.jackpots;
}

function simResetStats() {
    simStats = { plays: 0, spent: 0, won: 0, bestMatch: 0, jackpots: 0 };
    updateSimStats();
    showToast('Đã reset thống kê', 'info');
}

// ============================================
// Utilities
// ============================================
function showLoading(t) { document.getElementById('loadingText').textContent = t||'Đang xử lý...'; document.getElementById('loadingOverlay').classList.remove('hidden'); }
function hideLoading() { document.getElementById('loadingOverlay').classList.add('hidden'); }
function showToast(msg, type='info') { const t = document.getElementById('toast'); t.textContent = msg; t.className = `toast ${type} show`; setTimeout(()=>t.classList.remove('show'), 3000); }
