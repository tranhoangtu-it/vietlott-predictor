/**
 * Vietlott AI Predictor - Frontend Application
 * Static mode: loads pre-generated JSON data (no backend required)
 */

const DATA_BASE = './data';  // Static JSON files
let currentGame = 'power655';
let currentPage = 1;
let totalPages = 1;
let predictionsData = null;
let analysisData = null;
let historyCache = {};

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initGameSelect();
    initButtons();
    loadDashboard();
});

function initNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const page = link.dataset.page;
            switchPage(page);
        });
    });
}

function switchPage(page) {
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));

    document.querySelector(`[data-page="${page}"]`).classList.add('active');
    document.getElementById(`page-${page}`).classList.add('active');

    switch (page) {
        case 'dashboard': loadDashboard(); break;
        case 'predictions': loadPredictions(); break;
        case 'history': loadHistory(); break;
        case 'analysis': loadAnalysis(); break;
    }
}

function initGameSelect() {
    const select = document.getElementById('gameSelect');
    select.addEventListener('change', () => {
        currentGame = select.value;
        currentPage = 1;
        predictionsData = null;
        analysisData = null;
        const activePage = document.querySelector('.nav-link.active').dataset.page;
        switchPage(activePage);
    });
}

function initButtons() {
    document.getElementById('btnTrain').addEventListener('click', () => {
        showToast('Predictions được cập nhật tự động bởi GitHub Actions mỗi ngày!', 'info');
    });
    document.getElementById('btnRefreshPred').addEventListener('click', () => {
        predictionsData = null;
        loadDashboard();
    });
    document.getElementById('btnNewPredict').addEventListener('click', () => {
        predictionsData = null;
        loadPredictions();
    });
    document.getElementById('btnPrevPage').addEventListener('click', () => {
        if (currentPage > 1) { currentPage--; loadHistory(); }
    });
    document.getElementById('btnNextPage').addEventListener('click', () => {
        if (currentPage < totalPages) { currentPage++; loadHistory(); }
    });
}

// ============================================
// Data Loading (static JSON)
// ============================================
async function fetchJSON(url) {
    try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        return await resp.json();
    } catch (err) {
        console.error(`Fetch Error (${url}):`, err);
        return null;
    }
}

async function loadPredictionsData() {
    if (predictionsData) return predictionsData;

    const allPreds = await fetchJSON(`${DATA_BASE}/predictions.json`);
    if (allPreds && allPreds[currentGame]) {
        predictionsData = allPreds[currentGame];
    }
    return predictionsData;
}

async function loadAnalysisData() {
    if (analysisData) return analysisData;

    analysisData = await fetchJSON(`${DATA_BASE}/${currentGame}_analysis.json`);
    return analysisData;
}

async function loadHistoryData() {
    if (historyCache[currentGame]) return historyCache[currentGame];

    const data = await fetchJSON(`${DATA_BASE}/${currentGame}_history.json`);
    if (data) historyCache[currentGame] = data;
    return data;
}

// ============================================
// Dashboard
// ============================================
async function loadDashboard() {
    const preds = await loadPredictionsData();
    if (preds) {
        renderDashboard(preds);
    } else {
        document.getElementById('topPrediction').innerHTML =
            '<p style="text-align:center;color:var(--text-muted)">Chưa có dữ liệu. Hệ thống sẽ tự cập nhật bởi GitHub Actions.</p>';
    }

    // Also load analysis for frequency chart
    const analysis = await loadAnalysisData();
    if (analysis) {
        analysisData = analysis;
        renderHotCold(analysis);
        renderFrequencyChart(analysis);
    }
}

function renderDashboard(data) {
    // Stats
    document.getElementById('totalDraws').textContent = data.total_draws || '--';
    document.getElementById('lastUpdate').textContent = data.generated_at
        ? new Date(data.generated_at).toLocaleString('vi-VN')
        : '--';

    // Average confidence
    const allPreds = [
        ...(data.predictions?.ensemble || []),
        ...(data.predictions?.lstm || []),
        ...(data.predictions?.ml || []),
        ...(data.predictions?.statistical || []),
    ];
    if (allPreds.length > 0) {
        const avgConf = allPreds.reduce((s, p) => s + (p.confidence || 0), 0) / allPreds.length;
        document.getElementById('avgConfidence').textContent = avgConf.toFixed(1) + '%';
    }

    renderTopPredictions(data.predictions);
    renderRecentDraws(data.recent_draws || []);

    if (data.analysis) {
        analysisData = data.analysis;
        renderHotCold(data.analysis);
        renderFrequencyChart(data.analysis);
    }
}

function renderTopPredictions(predictions) {
    const container = document.getElementById('topPrediction');
    if (!predictions) {
        container.innerHTML = '<p style="text-align:center;color:var(--text-muted)">Chưa có dự đoán</p>';
        return;
    }

    let html = '';
    const sources = [
        { key: 'ensemble', label: 'Ensemble AI', icon: 'fas fa-layer-group' },
        { key: 'lstm', label: 'LSTM Neural Network', icon: 'fas fa-brain' },
        { key: 'ml', label: 'Random Forest + GB', icon: 'fas fa-tree' },
        { key: 'statistical', label: 'Statistical', icon: 'fas fa-chart-pie' },
    ];

    sources.forEach(source => {
        const preds = predictions[source.key];
        if (!preds || preds.length === 0) return;

        const best = preds[0];
        const confClass = best.confidence > 40 ? 'confidence-high' :
                         best.confidence > 25 ? 'confidence-mid' : 'confidence-low';

        html += `
            <div class="prediction-set">
                <div class="pred-header">
                    <span class="pred-method"><i class="${source.icon}"></i> ${source.label}</span>
                    <span class="pred-confidence ${confClass}">
                        ${best.confidence?.toFixed(1) || '?'}% tin cậy
                    </span>
                </div>
                <div class="pred-numbers">
                    ${best.numbers.map(n => `<span class="ball">${n}</span>`).join('')}
                </div>
            </div>
        `;
    });

    container.innerHTML = html || '<p style="text-align:center;color:var(--text-muted)">Chưa có dự đoán</p>';
}

function renderRecentDraws(draws) {
    const container = document.getElementById('recentDraws');
    if (!draws || draws.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">Không có dữ liệu</p>';
        return;
    }

    let html = '<table style="width:100%;border-collapse:collapse">';
    html += '<tr><th style="text-align:left;padding:8px;color:var(--text-muted);font-size:0.8rem">#</th>';
    html += '<th style="text-align:left;padding:8px;color:var(--text-muted);font-size:0.8rem">Kết quả</th></tr>';

    draws.slice(0, 8).reverse().forEach((draw, i) => {
        html += `<tr>
            <td style="padding:8px;font-size:0.85rem;color:var(--text-muted)">${draws.length - i}</td>
            <td style="padding:8px">
                ${draw.map(n => `<span class="ball ball-sm">${n}</span>`).join(' ')}
            </td>
        </tr>`;
    });

    html += '</table>';
    container.innerHTML = html;
}

function renderHotCold(analysis) {
    const container = document.getElementById('hotCold');
    if (!analysis) return;

    let html = '';

    if (analysis.hot_numbers) {
        html += `
            <div class="hot-cold-section">
                <h3><i class="fas fa-fire" style="color:#ef4444"></i> Số nóng (xuất hiện nhiều)</h3>
                <div class="balls-wrap">
                    ${analysis.hot_numbers.slice(0, 10).map(n =>
                        `<span class="ball ball-sm hot">${n}</span>`
                    ).join('')}
                </div>
            </div>
        `;
    }

    if (analysis.cold_numbers) {
        html += `
            <div class="hot-cold-section">
                <h3><i class="fas fa-snowflake" style="color:#06b6d4"></i> Số lạnh (xuất hiện ít)</h3>
                <div class="balls-wrap">
                    ${analysis.cold_numbers.slice(0, 10).map(n =>
                        `<span class="ball ball-sm cold">${n}</span>`
                    ).join('')}
                </div>
            </div>
        `;
    }

    if (analysis.overdue_numbers) {
        html += `
            <div class="hot-cold-section">
                <h3><i class="fas fa-clock" style="color:#8b5cf6"></i> Số lâu chưa ra (20 kỳ gần nhất)</h3>
                <div class="balls-wrap">
                    ${analysis.overdue_numbers.slice(0, 15).map(n =>
                        `<span class="ball ball-sm overdue">${n}</span>`
                    ).join('')}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function renderFrequencyChart(analysis) {
    const container = document.getElementById('frequencyChart');
    if (!analysis || !analysis.frequency) return;

    const freq = analysis.frequency;
    const numbers = Object.keys(freq).map(Number).sort((a, b) => a - b);
    const maxCount = Math.max(...numbers.map(n => freq[n].count));

    let html = '<div class="freq-bar-container">';
    numbers.forEach(n => {
        const count = freq[n].count;
        const pct = maxCount > 0 ? (count / maxCount) * 100 : 0;
        const isHot = analysis.hot_numbers?.includes(n);
        const isCold = analysis.cold_numbers?.includes(n);
        const color = isHot ? 'linear-gradient(90deg, #ef4444, #f97316)' :
                      isCold ? 'linear-gradient(90deg, #06b6d4, #3b82f6)' :
                      'linear-gradient(90deg, #6366f1, #8b5cf6)';

        html += `
            <div class="freq-bar-row">
                <span class="freq-bar-label">${n}</span>
                <div class="freq-bar-track">
                    <div class="freq-bar-fill" style="width:${pct}%;background:${color}"></div>
                </div>
                <span class="freq-bar-value">${count} (${freq[n].percentage}%)</span>
            </div>
        `;
    });
    html += '</div>';

    container.innerHTML = html;
}

// ============================================
// Predictions Page
// ============================================
async function loadPredictions() {
    const data = await loadPredictionsData();
    if (!data) {
        document.getElementById('allPredictions').innerHTML =
            '<p style="text-align:center;color:var(--text-muted)">Chưa có dữ liệu.</p>';
        return;
    }

    renderAllPredictions(data.predictions);
    renderMethodComparison();
}

function renderAllPredictions(predictions) {
    const container = document.getElementById('allPredictions');
    if (!predictions) return;

    let html = '';
    const methodConfig = {
        ensemble: { tag: 'tag-ensemble', label: 'Ensemble AI' },
        lstm: { tag: 'tag-lstm', label: 'LSTM Neural Network' },
        ml: { tag: 'tag-ml', label: 'Random Forest + GB' },
        statistical: { tag: 'tag-stat', label: 'Statistical Analysis' },
    };

    Object.entries(predictions).forEach(([method, preds]) => {
        if (!preds || preds.length === 0) return;
        const config = methodConfig[method] || { tag: '', label: method };

        preds.forEach((pred, i) => {
            const confClass = pred.confidence > 40 ? 'confidence-high' :
                             pred.confidence > 25 ? 'confidence-mid' : 'confidence-low';
            html += `
                <div class="prediction-card">
                    <span class="method-tag ${config.tag}">${config.label} #${i + 1}</span>
                    <div style="margin-bottom:8px">
                        <span class="pred-confidence ${confClass}" style="font-size:0.8rem">
                            ${pred.confidence?.toFixed(1) || '?'}% tin cậy
                        </span>
                    </div>
                    <div class="pred-numbers">
                        ${pred.numbers.map(n => `<span class="ball">${n}</span>`).join('')}
                    </div>
                </div>
            `;
        });
    });

    container.innerHTML = html || '<p style="text-align:center;color:var(--text-muted)">Chưa có dự đoán</p>';
}

function renderMethodComparison() {
    const container = document.getElementById('methodComparison');
    const methods = [
        {
            icon: 'fas fa-layer-group',
            name: 'Ensemble AI',
            desc: 'Kết hợp tất cả mô hình với trọng số: LSTM (40%) + RF/GB (35%) + Thống kê (25%). Cho kết quả ổn định nhất.',
            color: '#ec4899'
        },
        {
            icon: 'fas fa-brain',
            name: 'LSTM Neural Network',
            desc: 'Mạng nơ-ron hồi quy học chuỗi thời gian, phát hiện pattern ẩn trong dữ liệu lịch sử.',
            color: '#6366f1'
        },
        {
            icon: 'fas fa-tree',
            name: 'Random Forest + Gradient Boosting',
            desc: 'Mô hình học máy phân tích đặc trưng thống kê: tần suất, khoảng cách, xu hướng chẵn/lẻ.',
            color: '#10b981'
        },
        {
            icon: 'fas fa-chart-pie',
            name: 'Statistical Analysis',
            desc: 'Phân tích thống kê: số nóng/lạnh, cặp số phổ biến, phân bố tổng, tỷ lệ chẵn/lẻ.',
            color: '#f59e0b'
        }
    ];

    container.innerHTML = methods.map(m => `
        <div class="method-card">
            <h3><i class="${m.icon}" style="color:${m.color}"></i> ${m.name}</h3>
            <p>${m.desc}</p>
        </div>
    `).join('');
}

// ============================================
// History Page
// ============================================
async function loadHistory() {
    const allHistory = await loadHistoryData();
    if (!allHistory || allHistory.length === 0) {
        document.getElementById('historyTable').innerHTML =
            '<p style="text-align:center;color:var(--text-muted)">Chưa có dữ liệu lịch sử</p>';
        return;
    }

    const perPage = 20;
    totalPages = Math.ceil(allHistory.length / perPage);
    if (currentPage > totalPages) currentPage = totalPages;

    document.getElementById('pageInfo').textContent = `${currentPage} / ${totalPages}`;

    const start = (currentPage - 1) * perPage;
    const pageData = allHistory.slice(start, start + perPage);

    renderHistory(pageData);
}

function renderHistory(draws) {
    const container = document.getElementById('historyTable');

    let html = `<table>
        <thead>
            <tr>
                <th>Kỳ</th>
                <th>Ngày</th>
                <th>Kết quả</th>
                ${currentGame === 'power655' ? '<th>Power</th>' : ''}
                <th>Tổng</th>
            </tr>
        </thead>
        <tbody>`;

    draws.forEach(draw => {
        const numbers = draw.numbers || [];
        const sum = numbers.reduce((s, n) => s + (typeof n === 'number' ? n : parseInt(n) || 0), 0);

        html += `<tr>
            <td style="font-weight:600;color:var(--primary)">${draw.draw_id || '--'}</td>
            <td style="color:var(--text-muted);font-size:0.9rem">${draw.date || '--'}</td>
            <td>
                ${numbers.map(n => `<span class="ball ball-sm">${n}</span>`).join(' ')}
            </td>
            ${currentGame === 'power655' ? `<td><span class="ball ball-sm power">${draw.power || '--'}</span></td>` : ''}
            <td style="font-weight:600">${sum}</td>
        </tr>`;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ============================================
// Analysis Page
// ============================================
async function loadAnalysis() {
    const data = await loadAnalysisData();
    if (!data) return;

    renderOddEvenChart(data);
    renderDecadeChart(data);
    renderPairs(data);
    renderSumStats(data);
    renderOverdueNumbers(data);
}

function renderOddEvenChart(analysis) {
    const container = document.getElementById('oddEvenChart');
    if (!analysis.odd_even) return;

    const data = analysis.odd_even;
    let html = '<div style="display:flex;flex-direction:column;gap:8px;padding:10px">';
    Object.entries(data).forEach(([odd_count, pct]) => {
        const even_count = 6 - parseInt(odd_count);
        html += `
            <div style="display:flex;align-items:center;gap:12px">
                <span style="width:80px;font-size:0.85rem;color:var(--text-muted)">${odd_count}L/${even_count}C</span>
                <div style="flex:1;height:28px;background:rgba(255,255,255,0.05);border-radius:6px;overflow:hidden;position:relative">
                    <div style="height:100%;width:${pct}%;background:var(--gradient-1);border-radius:6px;display:flex;align-items:center;justify-content:flex-end;padding-right:8px">
                        <span style="font-size:0.75rem;font-weight:600;color:white">${pct}%</span>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderDecadeChart(analysis) {
    const container = document.getElementById('decadeChart');
    if (!analysis.decade_distribution) return;

    const data = analysis.decade_distribution;
    const maxVal = Math.max(...Object.values(data));

    let html = '<div style="display:flex;flex-direction:column;gap:8px;padding:10px">';
    Object.entries(data).forEach(([range, count]) => {
        const pct = maxVal > 0 ? (count / maxVal) * 100 : 0;
        html += `
            <div style="display:flex;align-items:center;gap:12px">
                <span style="width:60px;font-size:0.85rem;color:var(--text-muted);text-align:right">${range}</span>
                <div style="flex:1;height:28px;background:rgba(255,255,255,0.05);border-radius:6px;overflow:hidden">
                    <div style="height:100%;width:${pct}%;background:var(--gradient-3);border-radius:6px;display:flex;align-items:center;padding-left:8px">
                        <span style="font-size:0.75rem;font-weight:600;color:white">${count}</span>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    container.innerHTML = html;
}

function renderPairs(analysis) {
    const container = document.getElementById('pairsTable');
    if (!analysis.pairs) return;

    container.innerHTML = analysis.pairs.slice(0, 15).map(p => `
        <div class="pair-item">
            <div class="pair-numbers">${p.pair[0]} - ${p.pair[1]}</div>
            <div class="pair-count">${p.count} lần xuất hiện</div>
        </div>
    `).join('');
}

function renderSumStats(analysis) {
    const container = document.getElementById('sumStats');
    if (!analysis.sum_range) return;

    const stats = analysis.sum_range;
    container.innerHTML = `
        <div class="sum-stat-item">
            <div class="value">${stats.mean}</div>
            <div class="label">Trung bình</div>
        </div>
        <div class="sum-stat-item">
            <div class="value">${stats.median}</div>
            <div class="label">Trung vị</div>
        </div>
        <div class="sum-stat-item">
            <div class="value">${stats.std}</div>
            <div class="label">Độ lệch chuẩn</div>
        </div>
        <div class="sum-stat-item">
            <div class="value">${stats.min}</div>
            <div class="label">Tổng nhỏ nhất</div>
        </div>
        <div class="sum-stat-item">
            <div class="value">${stats.max}</div>
            <div class="label">Tổng lớn nhất</div>
        </div>
    `;
}

function renderOverdueNumbers(analysis) {
    const container = document.getElementById('overdueNumbers');
    if (!analysis.overdue_numbers) return;

    container.innerHTML = analysis.overdue_numbers.map(n =>
        `<span class="ball ball-sm overdue">${n}</span>`
    ).join('');
}

// ============================================
// Utilities
// ============================================
function showLoading(text) {
    document.getElementById('loadingText').textContent = text || 'Đang xử lý...';
    document.getElementById('loadingOverlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
}

function showToast(message, type = 'info') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3000);
}
