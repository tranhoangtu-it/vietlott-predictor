/**
 * AI Auto-Play Tab - Backtest AI predictions against real lottery data
 * Depends on globals from app.js: DATA_BASE, currentGame, fetchJSON, isDigitGame, showToast
 * LOAD ORDER: This file must load BEFORE app.js (which references autoplayData, loadAutoplay, etc.)
 */

// HTML-escape helper to prevent XSS from external data
function esc(str) {
    const d = document.createElement('div');
    d.textContent = String(str ?? '');
    return d.innerHTML;
}

let autoplayData = null;
let apCurrentPage = 1;
let apTotalPages = 1;

async function loadAutoplayData() {
    if (autoplayData) return autoplayData;
    autoplayData = await fetchJSON(`${DATA_BASE}/${currentGame}_autoplay.json`);
    return autoplayData;
}

async function loadAutoplay() {
    const data = await loadAutoplayData();
    if (!data || !data.plays) {
        document.getElementById('autoplayTable').innerHTML =
            '<p style="text-align:center;color:var(--text-muted)">Chưa có dữ liệu AI chơi cho game này.</p>';
        // Clear summary stats
        ['apTotalPlays','apTotalSpent','apTotalWon','apProfit','apROI','apWinRate','apBestMatch'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '--';
        });
        document.getElementById('apStrategy').textContent = '--';
        return;
    }
    renderAutoplaySummary(data);

    const pp = 15; // plays per page
    const plays = data.plays.slice().reverse(); // newest first
    apTotalPages = Math.ceil(plays.length / pp);
    if (apCurrentPage > apTotalPages) apCurrentPage = apTotalPages;
    document.getElementById('apPageInfo').textContent = `${apCurrentPage}/${apTotalPages}`;
    renderAutoplayTable(plays.slice((apCurrentPage - 1) * pp, apCurrentPage * pp));
}

function renderAutoplaySummary(data) {
    document.getElementById('apTotalPlays').textContent = data.total_plays?.toLocaleString() || '--';
    document.getElementById('apTotalSpent').textContent = formatVND(data.total_spent);
    document.getElementById('apTotalWon').textContent = formatVND(data.total_won);

    const profitEl = document.getElementById('apProfit');
    const profit = data.profit || 0;
    profitEl.textContent = (profit >= 0 ? '+' : '') + formatVND(profit);
    profitEl.style.color = profit >= 0 ? '#10b981' : '#ef4444';

    const roiEl = document.getElementById('apROI');
    roiEl.textContent = (data.roi >= 0 ? '+' : '') + data.roi + '%';
    roiEl.style.color = data.roi >= 0 ? '#10b981' : '#ef4444';

    document.getElementById('apWinRate').textContent = data.win_rate + '%';
    document.getElementById('apBestMatch').textContent = data.best_match || 0;
    document.getElementById('apStrategy').textContent = 'Statistical AI';
}

function renderAutoplayTable(plays) {
    const c = document.getElementById('autoplayTable');
    if (!plays?.length) {
        c.innerHTML = '<p style="text-align:center;color:var(--text-muted)">Không có dữ liệu.</p>';
        return;
    }

    let html = `<div class="history-table"><table><thead><tr>
        <th>Kỳ</th><th>Ngày</th><th>AI Chọn</th><th>Kết Quả</th>
        <th>Trùng</th><th>Thưởng</th><th>Lũy kế</th>
    </tr></thead><tbody>`;

    plays.forEach(p => {
        const aiNums = p.ai_numbers || [];
        const actualNums = p.actual_numbers || [];
        const actualSet = new Set(actualNums.map(String));
        const isWin = p.prize > 0;

        html += `<tr class="${isWin ? 'ap-row-win' : ''}">
            <td style="font-weight:600;color:var(--primary-light);font-family:'JetBrains Mono',monospace;font-size:0.85rem">${esc(p.draw_id) || '--'}</td>
            <td style="color:var(--text-muted);font-size:0.85rem">${esc(p.date) || '--'}</td>
            <td>${aiNums.map(n => {
                const matched = actualSet.has(String(n));
                return `<span class="ball ball-sm ${matched ? 'hot' : ''}">${esc(n)}</span>`;
            }).join(' ')}</td>
            <td>${actualNums.map(n => `<span class="ball ball-sm">${esc(n)}</span>`).join(' ')}${p.actual_power != null ? ` <span class="ball ball-sm power">${esc(p.actual_power)}</span>` : ''}</td>
            <td class="ap-match-cell ${isWin ? 'win' : 'lose'}">${esc(p.matches)}/${aiNums.length}</td>
            <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;${isWin ? 'color:#10b981;font-weight:600' : 'color:var(--text-muted)'}">${isWin ? formatVND(p.prize) : '0'}</td>
            <td style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;font-weight:600;color:${p.cumulative_profit >= 0 ? '#10b981' : '#ef4444'}">${(p.cumulative_profit >= 0 ? '+' : '') + formatVND(p.cumulative_profit)}</td>
        </tr>`;
    });

    c.innerHTML = html + '</tbody></table></div>';
}

function formatVND(amount) {
    if (amount == null) return '--';
    return Math.abs(amount) >= 1e9
        ? (amount / 1e9).toFixed(1) + ' tỷ'
        : amount.toLocaleString('vi-VN');
}

// Next draw calculator - used by renderDashboard in app.js
function calculateNextDraw(schedule) {
    if (!schedule || !schedule.days) return '--';

    const now = new Date();
    const dayNames = ['CN', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'];

    if (schedule.frequency === '10min') {
        // Keno/Bingo: next draw is within minutes
        const hour = now.getHours();
        const min = now.getMinutes();
        if (hour >= 6 && (hour < 21 || (hour === 21 && min <= 55))) {
            const nextMin = Math.ceil(min / 10) * 10;
            const nextHour = nextMin >= 60 ? hour + 1 : hour;
            // Guard: if computed time exceeds draw window, show tomorrow
            if (nextHour > 21 || (nextHour === 21 && (nextMin % 60) > 55)) {
                return 'Ngày mai 06:00';
            }
            return `Hôm nay ${String(nextHour).padStart(2,'0')}:${String(nextMin % 60).padStart(2,'0')}`;
        }
        return 'Ngày mai 06:00';
    }

    // Daily games: find next draw day
    const drawTime = schedule.time || '21:00';
    const [drawH, drawM] = drawTime.split(':').map(Number);

    for (let offset = 0; offset < 7; offset++) {
        const candidate = new Date(now);
        candidate.setDate(candidate.getDate() + offset);
        const day = candidate.getDay();

        if (schedule.days.includes(day)) {
            // If today, check if draw time hasn't passed
            if (offset === 0) {
                if (now.getHours() > drawH || (now.getHours() === drawH && now.getMinutes() >= drawM)) continue;
            }
            const dd = String(candidate.getDate()).padStart(2, '0');
            const mm = String(candidate.getMonth() + 1).padStart(2, '0');
            return `${dayNames[day]} ${dd}/${mm} ${drawTime}`;
        }
    }
    return '--';
}

// Init pagination buttons for autoplay tab
function initAutoplayPagination() {
    const btnPrev = document.getElementById('btnApPrev');
    const btnNext = document.getElementById('btnApNext');
    if (btnPrev) btnPrev.addEventListener('click', () => {
        if (apCurrentPage > 1) { apCurrentPage--; loadAutoplay(); }
    });
    if (btnNext) btnNext.addEventListener('click', () => {
        if (apCurrentPage < apTotalPages) { apCurrentPage++; loadAutoplay(); }
    });
}
