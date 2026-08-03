/**
 * Climate Trend Analyzer - Chart.js Interactive Charts (v10.1 RC)
 * Executive-grade charts with consistent color mapping and light theme.
 * Temperature=Red/Orange, Precipitation=Blue, Humidity=Teal, Solar=Amber,
 * Forecast=Purple/Amber, Anomalies=Red.
 */

const valueLabelPlugin = {
    id: 'valueLabels',
    afterDatasetsDraw(chart) {
        const { ctx } = chart;
        chart.data.datasets.forEach((dataset, i) => {
            const meta = chart.getDatasetMeta(i);
            if (meta.hidden) return;
            meta.data.forEach((bar, index) => {
                const value = dataset.data[index];
                if (value === null || value === undefined) return;
                ctx.save();
                ctx.fillStyle = '#4A5568';
                ctx.font = "600 10px 'Segoe UI', sans-serif";
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillText(value.toFixed(1) + '\u00B0', bar.x, bar.y - 6);
                ctx.restore();
            });
        });
    }
};

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: true,
    animation: { duration: 600, easing: 'easeOutQuart' },
    plugins: {
        legend: {
            display: true,
            position: 'top',
            align: 'end',
            labels: {
                color: '#4A5568',
                font: { size: 11, family: "'Segoe UI', sans-serif", weight: 500 },
                usePointStyle: true,
                pointStyleWidth: 10,
                padding: 20,
                boxHeight: 8,
            }
        },
        tooltip: {
            enabled: true,
            backgroundColor: '#FFFFFF',
            titleColor: '#1A2332',
            bodyColor: '#4A5568',
            borderColor: '#E2E8F0',
            borderWidth: 1,
            cornerRadius: 10,
            padding: 14,
            boxPadding: 6,
            titleFont: { size: 12, weight: 700, family: "'Segoe UI', sans-serif" },
            bodyFont: { size: 11, family: "'Segoe UI', sans-serif" },
            displayColors: true,
            boxWidth: 10,
            boxHeight: 10,
            boxPadding: 4,
            caretSize: 6,
            caretPadding: 8,
        }
    },
    scales: {
        x: {
            ticks: {
                color: '#8494A7',
                font: { size: 10, family: "'Segoe UI', sans-serif" },
                maxTicksLimit: 10,
                maxRotation: 0,
                padding: 6,
            },
            grid: { color: '#F0F2F7', lineWidth: 1 },
            border: { color: '#E2E8F0', width: 1 }
        },
        y: {
            ticks: {
                color: '#8494A7',
                font: { size: 10, family: "'Segoe UI', sans-serif" },
                padding: 8,
            },
            grid: { color: '#F0F2F7', lineWidth: 1 },
            border: { color: '#E2E8F0', width: 1 }
        }
    },
    elements: {
        line: { borderWidth: 1.5, tension: 0.3 },
        point: { radius: 0, hoverRadius: 5, hoverBorderWidth: 2, hoverBackgroundColor: '#FFFFFF' }
    }
};

function axisLabelConfig(title) {
    return {
        display: true,
        text: title,
        color: '#8494A7',
        font: { size: 10, family: "'Segoe UI', sans-serif", weight: 500 },
        padding: { top: 10 }
    };
}

function yearCallback() {
    return function(val) {
        const label = this.getLabelForValue(val);
        return label && label.endsWith('-01-01') ? label.split('-')[0] : '';
    };
}

async function loadTemperatureChart() {
    const data = await fetchJSON('daily_trends.json');
    if (!data || data.length === 0) return;

    const labels = data.map(d => d.date);
    const temps = data.map(d => d.temperature);

    const ma30 = [];
    for (let i = 0; i < temps.length; i++) {
        if (i < 29) { ma30.push(null); continue; }
        const slice = temps.slice(i - 29, i + 1);
        ma30.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }

    new Chart(document.getElementById('chart-temperature'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Daily Temperature',
                    data: temps,
                    borderColor: '#E67E22',
                    backgroundColor: 'rgba(230, 126, 34, 0.04)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: '30-Day Moving Avg',
                    data: ma30,
                    borderColor: '#DC3545',
                    borderWidth: 2.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.3,
                }
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: {
                    ...chartDefaults.scales.x,
                    ticks: { ...chartDefaults.scales.x.ticks, callback: yearCallback() }
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: axisLabelConfig('Temperature (\u00B0C)')
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: ctx => ctx.parsed.y !== null ? `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} \u00B0C` : ''
                    }
                }
            }
        }
    });
}

async function loadPrecipitationChart() {
    const data = await fetchJSON('daily_trends.json');
    if (!data || data.length === 0) return;

    const labels = data.map(d => d.date);
    const precip = data.map(d => d.precipitation);

    new Chart(document.getElementById('chart-precipitation'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Precipitation',
                data: precip,
                backgroundColor: 'rgba(59, 130, 246, 0.3)',
                borderColor: 'rgba(59, 130, 246, 0.5)',
                borderWidth: 0,
                borderRadius: 2,
                barPercentage: 0.9,
                categoryPercentage: 0.95,
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: {
                    ...chartDefaults.scales.x,
                    ticks: { ...chartDefaults.scales.x.ticks, callback: yearCallback() }
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: axisLabelConfig('Precipitation (mm/day)')
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: ctx => `${ctx.parsed.y.toFixed(2)} mm/day`
                    }
                }
            }
        }
    });
}

async function loadForecastChart() {
    const historical = await fetchJSON('daily_trends.json');
    const forecast = await fetchJSON('forecast.json');

    const datasets = [];

    if (historical && historical.length > 0) {
        datasets.push({
            label: 'Historical Temperature',
            data: historical.map(d => ({ x: d.date, y: d.temperature })),
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59, 130, 246, 0.03)',
            borderWidth: 1.5,
            fill: true,
            pointRadius: 0,
            tension: 0.3,
        });
    }

    if (forecast && forecast.length > 0) {
        datasets.push({
            label: 'Forecast',
            data: forecast.map(d => ({ x: d.date, y: d.forecast })),
            borderColor: '#8B5CF6',
            borderWidth: 2.5,
            borderDash: [6, 3],
            fill: false,
            pointRadius: 0,
            tension: 0.3,
        });
        datasets.push({
            label: 'Upper Bound (95% CI)',
            data: forecast.map(d => ({ x: d.date, y: d.forecast_upper })),
            borderColor: 'rgba(139, 92, 246, 0.2)',
            backgroundColor: 'rgba(139, 92, 246, 0.05)',
            borderWidth: 1,
            fill: '+1',
            pointRadius: 0,
            tension: 0.3,
        });
        datasets.push({
            label: 'Lower Bound (95% CI)',
            data: forecast.map(d => ({ x: d.date, y: d.forecast_lower })),
            borderColor: 'rgba(139, 92, 246, 0.2)',
            borderWidth: 1,
            fill: false,
            pointRadius: 0,
            tension: 0.3,
        });
    }

    const forecastStart = forecast && forecast.length > 0 ? forecast[0].date : null;

    new Chart(document.getElementById('chart-forecast'), {
        type: 'line',
        data: { datasets },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: {
                    type: 'category',
                    ticks: {
                        color: '#8494A7',
                        font: { size: 10, family: "'Segoe UI', sans-serif" },
                        maxTicksLimit: 12,
                        maxRotation: 0,
                        callback: function(val) {
                            const label = this.getLabelForValue(val);
                            return label && (label.endsWith('-01-01') || label === forecastStart)
                                ? (label === forecastStart ? 'Forecast \u2192' : label.split('-')[0])
                                : '';
                        }
                    },
                    grid: { color: '#F0F2F7', lineWidth: 1 },
                    border: { color: '#E2E8F0', width: 1 }
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: axisLabelConfig('Temperature (\u00B0C)')
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                legend: {
                    ...chartDefaults.plugins.legend,
                    position: 'top',
                    align: 'center',
                },
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        title: items => items[0]?.label || '',
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} \u00B0C`
                    }
                },
                annotation: forecastStart ? {
                    annotations: {
                        forecastLine: {
                            type: 'line',
                            xMin: forecastStart,
                            xMax: forecastStart,
                            borderColor: 'rgba(139, 92, 246, 0.5)',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            label: {
                                display: true,
                                content: 'Forecast Start',
                                position: 'start',
                                color: '#8B5CF6',
                                font: { size: 10, weight: 600 },
                                backgroundColor: 'rgba(255,255,255,0.9)',
                                padding: { top: 4, bottom: 4, left: 8, right: 8 },
                                borderRadius: 4,
                            }
                        }
                    }
                } : undefined
            }
        }
    });
}

async function loadAnomalyChart() {
    const daily = await fetchJSON('daily_trends.json');
    const anomalies = await fetchJSON('anomalies.json');

    if (!daily || daily.length === 0) return;

    const normalData = daily.map(d => ({ x: d.date, y: d.temperature }));
    const anomalyData = (anomalies || []).map(d => ({ x: d.date, y: d.temperature }));

    new Chart(document.getElementById('chart-anomalies'), {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Normal Observations',
                    data: normalData,
                    backgroundColor: 'rgba(59, 130, 246, 0.12)',
                    pointRadius: 1.5,
                    pointHoverRadius: 4,
                },
                {
                    label: 'Anomaly',
                    data: anomalyData,
                    backgroundColor: '#DC3545',
                    pointRadius: 4,
                    pointHoverRadius: 7,
                    pointHoverBorderWidth: 2,
                    pointHoverBorderColor: '#FFFFFF',
                }
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    type: 'category',
                    ticks: {
                        color: '#8494A7',
                        font: { size: 10, family: "'Segoe UI', sans-serif" },
                        maxTicksLimit: 10,
                        maxRotation: 0,
                        callback: yearCallback()
                    },
                    grid: { color: '#F0F2F7', lineWidth: 1 },
                    border: { color: '#E2E8F0', width: 1 }
                },
                y: {
                    ticks: { color: '#8494A7', font: { size: 10, family: "'Segoe UI', sans-serif" }, padding: 8 },
                    grid: { color: '#F0F2F7', lineWidth: 1 },
                    border: { color: '#E2E8F0', width: 1 },
                    title: axisLabelConfig('Temperature (\u00B0C)')
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} \u00B0C on ${ctx.label}`
                    }
                }
            }
        }
    });
}

async function loadHumidityChart() {
    const data = await fetchJSON('daily_trends.json');
    if (!data || data.length === 0) return;

    const labels = data.map(d => d.date);
    const humidity = data.map(d => d.humidity);

    const ma30 = [];
    for (let i = 0; i < humidity.length; i++) {
        if (i < 29) { ma30.push(null); continue; }
        const slice = humidity.slice(i - 29, i + 1);
        ma30.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }

    new Chart(document.getElementById('chart-humidity'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Daily Humidity',
                    data: humidity,
                    borderColor: '#0D9488',
                    backgroundColor: 'rgba(13, 148, 136, 0.04)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: '30-Day Moving Avg',
                    data: ma30,
                    borderColor: '#DC3545',
                    borderWidth: 2.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.3,
                }
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: {
                    ...chartDefaults.scales.x,
                    ticks: { ...chartDefaults.scales.x.ticks, callback: yearCallback() }
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: axisLabelConfig('Relative Humidity (%)'),
                    min: 0,
                    max: 100,
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: ctx => ctx.parsed.y !== null ? `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}%` : ''
                    }
                }
            }
        }
    });
}

async function loadSolarChart() {
    const data = await fetchJSON('daily_trends.json');
    if (!data || data.length === 0) return;

    const labels = data.map(d => d.date);
    const solar = data.map(d => d.solar_radiation);

    const ma30 = [];
    for (let i = 0; i < solar.length; i++) {
        if (i < 29) { ma30.push(null); continue; }
        const slice = solar.slice(i - 29, i + 1);
        ma30.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }

    new Chart(document.getElementById('chart-solar'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                {
                    label: 'Daily Solar Radiation',
                    data: solar,
                    borderColor: '#F59E0B',
                    backgroundColor: 'rgba(245, 158, 11, 0.04)',
                    borderWidth: 1,
                    fill: true,
                    pointRadius: 0,
                    tension: 0.2,
                },
                {
                    label: '30-Day Moving Avg',
                    data: ma30,
                    borderColor: '#DC3545',
                    borderWidth: 2.5,
                    fill: false,
                    pointRadius: 0,
                    tension: 0.3,
                }
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: {
                    ...chartDefaults.scales.x,
                    ticks: { ...chartDefaults.scales.x.ticks, callback: yearCallback() }
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: axisLabelConfig('Solar Radiation (MJ/m\u00B2/day)')
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: ctx => ctx.parsed.y !== null ? `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} MJ/m\u00B2/day` : ''
                    }
                }
            }
        }
    });
}

async function loadMonthlyTempChart() {
    const data = await fetchJSON('daily_trends.json');
    if (!data || data.length === 0) return;

    const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const monthTotals = Array(12).fill(0);
    const monthCounts = Array(12).fill(0);

    data.forEach(d => {
        const month = new Date(d.date).getMonth();
        monthTotals[month] += d.temperature;
        monthCounts[month]++;
    });

    const monthAvg = monthTotals.map((t, i) => monthCounts[i] > 0 ? t / monthCounts[i] : 0);

    new Chart(document.getElementById('chart-monthly-temp'), {
        type: 'bar',
        data: {
            labels: monthNames,
            datasets: [{
                label: 'Average Temperature',
                data: monthAvg,
                backgroundColor: monthAvg.map(v => {
                    if (v >= 30) return 'rgba(220, 53, 69, 0.65)';
                    if (v >= 25) return 'rgba(230, 126, 34, 0.65)';
                    if (v >= 20) return 'rgba(245, 158, 11, 0.65)';
                    return 'rgba(59, 130, 246, 0.65)';
                }),
                borderColor: monthAvg.map(v => {
                    if (v >= 30) return '#DC3545';
                    if (v >= 25) return '#E67E22';
                    if (v >= 20) return '#F59E0B';
                    return '#3B82F6';
                }),
                borderWidth: 1,
                borderRadius: 6,
                barPercentage: 0.7,
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    title: axisLabelConfig('Temperature (\u00B0C)'),
                    beginAtZero: false,
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: ctx => `${ctx.label}: ${ctx.parsed.y.toFixed(2)} \u00B0C`
                    }
                },
                valueLabels: { display: true }
            }
        },
        plugins: [valueLabelPlugin]
    });
}
