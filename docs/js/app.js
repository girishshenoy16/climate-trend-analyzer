/**
 * Climate Trend Analyzer - Executive Dashboard Application
 * Fetches JSON feeds and populates the DOM with KPIs, risk assessment,
 * executive summary, insights, recommendations, and forecast metadata.
 */

const DATA_BASE = 'data/';

async function fetchJSON(filename) {
    try {
        const response = await fetch(DATA_BASE + filename);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();
    } catch (err) {
        console.warn(`Failed to load ${filename}:`, err.message);
        return null;
    }
}

function updateKPI(id, value, decimals = 1) {
    const el = document.getElementById(id);
    if (el && value !== null && value !== undefined) {
        el.textContent = typeof value === 'number' ? value.toFixed(decimals) : value;
    }
}

function formatTimestamp(isoString) {
    if (!isoString) return '';
    const d = new Date(isoString);
    const months = ['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'];
    return `Last updated: ${d.getDate()} ${months[d.getMonth()]} ${d.getFullYear()}`;
}

async function loadExecutiveSummary() {
    const data = await fetchJSON('executive_summary.json');
    if (!data) return;

    const kpis = data.kpis || {};

    updateKPI('kpi-avg-temp', kpis.avg_temperature);
    updateKPI('kpi-warming-rate', kpis.warming_rate_per_decade, 3);
    updateKPI('kpi-precipitation', kpis.avg_precipitation);
    updateKPI('kpi-anomaly-days', kpis.anomaly_days, 0);
    updateKPI('kpi-forecast-trend', kpis.forecast_trend_per_decade, 3);

    const tempTrend = document.getElementById('kpi-temp-trend');
    if (tempTrend) {
        const rate = kpis.warming_rate_per_decade || 0;
        tempTrend.textContent = rate > 0 ? 'Increasing' : rate < 0 ? 'Decreasing' : 'Stable';
        tempTrend.className = 'kpi-trend ' + (rate > 0 ? 'trend-up' : rate < 0 ? 'trend-down' : 'trend-stable');
    }

    const precipTrend = document.getElementById('kpi-precip-trend');
    if (precipTrend) {
        precipTrend.textContent = 'Variable';
        precipTrend.className = 'kpi-trend trend-stable';
    }

    const anomalyTrend = document.getElementById('kpi-anomaly-trend');
    if (anomalyTrend) {
        const pct = kpis.anomaly_percentage || 0;
        anomalyTrend.textContent = `${pct.toFixed(1)}%`;
        anomalyTrend.className = 'kpi-trend ' + (pct > 5 ? 'trend-up' : 'trend-stable');
    }

    const timestampEl = document.getElementById('last-updated');
    if (timestampEl && data.generated_at) {
        timestampEl.textContent = formatTimestamp(data.generated_at);
    }

    const stationEl = document.getElementById('hero-station');
    if (stationEl && kpis.station_name) {
        stationEl.textContent = kpis.station_name;
    }

    const periodEl = document.getElementById('hero-period');
    if (periodEl && kpis.analysis_start_year && kpis.analysis_end_year) {
        periodEl.textContent = `${kpis.analysis_start_year} \u2013 ${kpis.analysis_end_year}`;
    }

    const versionEl = document.getElementById('hero-version');
    if (versionEl && data.pipeline_version) {
        versionEl.textContent = `v${data.pipeline_version}`;
    }

    const modeEl = document.getElementById('hero-mode');
    if (modeEl) {
        modeEl.textContent = kpis.data_source === 'api' ? 'Live API' : 'Simulation';
    }

    const riskCat = kpis.risk_category || 'Unknown';
    const riskScore = kpis.risk_score || 0;

    const riskLevel = document.getElementById('risk-level');
    if (riskLevel) {
        riskLevel.textContent = riskCat;
        riskLevel.setAttribute('data-risk', riskCat);
    }

    const riskScoreValue = document.getElementById('risk-score-value');
    if (riskScoreValue) {
        riskScoreValue.textContent = riskScore.toFixed(2);
    }

    const confidenceValue = document.getElementById('risk-confidence-value');
    if (confidenceValue) {
        const reliability = kpis.forecast_reliability || 'N/A';
        const confidence = kpis.quality_checks_passed ? 'High' : kpis.forecast_reliability === 'High' ? 'Moderate' : 'Moderate';
        confidenceValue.textContent = confidence;
        const confColors = { High: '#28A745', Moderate: '#F59E0B', Low: '#E67E22' };
        confidenceValue.style.color = confColors[confidence] || '#1A2332';
    }

    const riskBadge = document.getElementById('kpi-risk-badge');
    if (riskBadge) {
        riskBadge.textContent = riskCat;
        riskBadge.className = 'kpi-trend';
        const colors = { Low: 'trend-down', Moderate: 'trend-stable', High: 'trend-up', 'Very High': 'trend-up' };
        riskBadge.classList.add(colors[riskCat] || 'trend-stable');
    }

    const riskEl = document.getElementById('kpi-risk-category');
    if (riskEl && kpis.risk_category) {
        riskEl.textContent = kpis.risk_category;
    }

    const riskScoreEl = document.getElementById('kpi-risk-score');
    if (riskScoreEl) {
        riskScoreEl.textContent = `Score: ${riskScore.toFixed(2)}`;
    }

    const breakdown = document.getElementById('risk-breakdown');
    if (breakdown && kpis.risk_components) {
        const components = [
            { label: 'Temp Trend', value: kpis.risk_components.temp_trend },
            { label: 'Rainfall Dev', value: kpis.risk_components.rainfall_deviation },
            { label: 'Anomaly Freq', value: kpis.risk_components.anomaly_frequency },
            { label: 'Forecast', value: kpis.risk_components.forecast_consistency },
        ];
        breakdown.innerHTML = components.map(c => {
            const pct = (c.value * 100).toFixed(0);
            const color = c.value >= 0.7 ? '#28A745' : c.value >= 0.4 ? '#F59E0B' : '#DC3545';
            return `
            <div class="risk-component">
                <span class="risk-component-value" style="color: ${color}">${pct}%</span>
                <span class="risk-component-label">${c.label}</span>
            </div>`;
        }).join('');
    }

    const interpretation = document.getElementById('risk-interpretation');
    if (interpretation && kpis.risk_category) {
        const interpMap = {
            'Low': 'Climate risk is low. The region shows stable patterns with minimal deviation from historical norms. Standard monitoring recommended.',
            'Moderate': 'Moderate climate risk detected. Some indicators show variability that warrants continued monitoring and adaptive planning.',
            'High': 'Elevated climate risk. Multiple indicators suggest significant deviation from baseline. Enhanced monitoring and mitigation strategies recommended.',
            'Very High': 'Critical climate risk level. Multiple stress factors detected. Immediate adaptive measures and comprehensive monitoring required.'
        };
        interpretation.textContent = interpMap[kpis.risk_category] || 'Risk assessment based on temperature trends, rainfall deviation, and anomaly frequency.';
    }

    const summaryGrid = document.getElementById('exec-summary-grid');
    if (summaryGrid) {
        const items = [
            { icon: '&#127777;', label: 'Climate Trend', value: `<strong>${kpis.warming_rate_per_decade?.toFixed(3) || 'N/A'}\u00B0C</strong> per decade over <strong>${kpis.total_years || 'N/A'}</strong> years` },
            { icon: '&#128202;', label: 'Forecast Confidence', value: `<strong>${kpis.forecast_reliability || 'N/A'}</strong> reliability (R\u00B2 = <strong>${kpis.model_r_squared?.toFixed(4) || 'N/A'}</strong>)` },
            { icon: '&#9888;', label: 'Anomaly Detection', value: `<strong>${kpis.anomaly_days || 0}</strong> anomaly days detected (<strong>${kpis.anomaly_percentage?.toFixed(1) || 'N/A'}%</strong> of period)` },
            { icon: '&#128200;', label: 'Overall Risk', value: `<strong>${riskCat}</strong> risk (score: <strong>${riskScore.toFixed(2)}</strong>) based on temperature, rainfall, and anomaly indicators` },
            { icon: '&#128196;', label: 'Model Performance', value: `RMSE: <strong>${kpis.validation_avg_rmse?.toFixed(2) || 'N/A'}</strong> | MAE: <strong>${kpis.validation_avg_mae?.toFixed(2) || 'N/A'}</strong> | MAPE: <strong>${kpis.validation_avg_mape?.toFixed(1) || 'N/A'}%</strong>` },
            { icon: '&#127919;', label: 'Recommendation', value: kpis.recommended_action || 'Continue standard monitoring.' },
        ];
        summaryGrid.innerHTML = items.map(item => `
            <div class="exec-summary-item">
                <span class="exec-summary-icon">${item.icon}</span>
                <div class="exec-summary-content">
                    <span class="exec-summary-label">${item.label}</span>
                    <span class="exec-summary-value">${item.value}</span>
                </div>
            </div>
        `).join('');
    }

    if (data.insights && data.insights.length > 0) {
        const banner = document.getElementById('insight-banner');
        const grid = document.getElementById('insight-grid');
        if (banner && grid) {
            banner.style.display = 'block';
            const icons = ['&#127777;', '&#127783;', '&#9888;', '&#128202;'];
            grid.innerHTML = data.insights.map((insight, i) => {
                const icon = icons[i % icons.length];
                return `<div class="insight-item">
                    <span class="insight-icon">${icon}</span>
                    <span class="insight-text">${insight}</span>
                </div>`;
            }).join('');
        }
    }

    if (data.recommendations && data.recommendations.length > 0) {
        const section = document.getElementById('recommendations-section');
        const container = document.getElementById('recommendations-list');
        if (section && container) {
            section.style.display = 'block';
            const high = [], medium = [], low = [];
            data.recommendations.forEach((rec, i) => {
                if (i === 0) high.push(rec);
                else if (i < 3) medium.push(rec);
                else low.push(rec);
            });
            let html = '';
            if (high.length > 0) {
                html += `<div class="rec-group"><div class="rec-group-label label-high">High Priority</div>`;
                html += high.map(rec => `<div class="rec-item"><span class="rec-priority priority-high">High</span><span>${rec}</span></div>`).join('');
                html += '</div>';
            }
            if (medium.length > 0) {
                html += `<div class="rec-group"><div class="rec-group-label label-medium">Medium Priority</div>`;
                html += medium.map(rec => `<div class="rec-item"><span class="rec-priority priority-medium">Medium</span><span>${rec}</span></div>`).join('');
                html += '</div>';
            }
            if (low.length > 0) {
                html += `<div class="rec-group"><div class="rec-group-label label-low">Low Priority</div>`;
                html += low.map(rec => `<div class="rec-item"><span class="rec-priority priority-low">Low</span><span>${rec}</span></div>`).join('');
                html += '</div>';
            }
            container.innerHTML = html;
        }
    }
}

async function loadDashboard() {
    await Promise.all([
        loadExecutiveSummary(),
        loadTemperatureChart(),
        loadPrecipitationChart(),
        loadHumidityChart(),
        loadSolarChart(),
        loadForecastChart(),
        loadAnomalyChart(),
        loadMonthlyTempChart(),
        loadForecastMeta(),
        initMap(),
    ]);
}

async function loadForecastMeta() {
    const forecast = await fetchJSON('forecast.json');
    if (!forecast || !forecast.metadata) return;

    const meta = forecast.metadata;
    const section = document.getElementById('forecast-meta');
    const grid = document.getElementById('forecast-meta-grid');
    if (!section || !grid) return;

    section.style.display = 'block';

    const reliabilityColors = { High: '#28A745', Moderate: '#F59E0B', Low: '#E67E22' };
    const relColor = reliabilityColors[meta.reliability_label] || '#1A2332';

    grid.innerHTML = `
        <div class="meta-card">
            <span class="meta-label">Reliability</span>
            <span class="meta-value" style="color: ${relColor}">${meta.reliability_label || 'N/A'}</span>
        </div>
        <div class="meta-card">
            <span class="meta-label">Score</span>
            <span class="meta-value">${(meta.reliability_score || 0).toFixed(3)}</span>
        </div>
        <div class="meta-card">
            <span class="meta-label">Trend</span>
            <span class="meta-value">${(meta.trend_per_decade || 0).toFixed(3)} \u00B0C/dec</span>
        </div>
        <div class="meta-card">
            <span class="meta-label">R\u00B2</span>
            <span class="meta-value">${(meta.trend_r_squared || 0).toFixed(4)}</span>
        </div>
        <div class="meta-card">
            <span class="meta-label">Classification</span>
            <span class="meta-value">${(meta.forecast_class || 'N/A').replace('_', ' ')}</span>
        </div>
    `;
}

document.addEventListener('DOMContentLoaded', loadDashboard);
