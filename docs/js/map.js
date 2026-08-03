/**
 * Climate Trend Analyzer - Leaflet.js Regional Map (v10.0)
 * Interactive map with climate station markers, rich popups,
 * and contextual metadata. Light theme with CartoDB tiles.
 */

let map = null;
let markersLayer = null;

function initMap() {
    if (map) return;

    map = L.map('climate-map', {
        center: [28.6139, 77.2090],
        zoom: 5,
        zoomControl: true,
        attributionControl: true,
        scrollWheelZoom: false,
    });

    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/">CARTO</a>',
        maxZoom: 19,
        subdomains: 'abcd',
    }).addTo(map);

    markersLayer = L.layerGroup().addTo(map);

    loadStationMarkers();
}

async function loadStationMarkers() {
    const data = await fetchJSON('regional_map.json');
    if (!data || !data.stations) return;

    const riskColors = {
        'Low': '#28A745',
        'Moderate': '#F59E0B',
        'High': '#E67E22',
        'Very High': '#DC3545',
    };

    const station = data.stations[0];

    if (station) {
        const nameEl = document.getElementById('map-station-name');
        const latEl = document.getElementById('map-latitude');
        const lonEl = document.getElementById('map-longitude');
        const zoneEl = document.getElementById('map-climate-zone');
        const covEl = document.getElementById('map-coverage');

        if (nameEl) nameEl.textContent = station.name || '--';
        if (latEl) latEl.textContent = `${station.lat.toFixed(4)}\u00B0N`;
        if (lonEl) lonEl.textContent = `${station.lon.toFixed(4)}\u00B0E`;
        if (zoneEl) zoneEl.textContent = station.climate_zone || 'Tropical Wet & Dry';
        if (covEl) covEl.textContent = '2015 \u2013 2024';
    }

    data.stations.forEach(s => {
        const color = riskColors[s.risk_category] || '#3B82F6';

        const icon = L.divIcon({
            className: 'custom-marker',
            html: `<div style="
                width: 24px; height: 24px;
                background: ${color};
                border: 3px solid white;
                border-radius: 50%;
                box-shadow: 0 2px 10px ${color}50;
                transition: transform 0.2s ease;
            "></div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12],
        });

        const marker = L.marker([s.lat, s.lon], { icon })
            .addTo(markersLayer);

        const popupContent = `
            <div style="font-family: 'Segoe UI', sans-serif; min-width: 240px; padding: 6px;">
                <h4 style="margin: 0 0 12px 0; color: #1A2332; font-size: 15px; font-weight: 700; border-bottom: 2px solid ${color}; padding-bottom: 8px;">
                    ${s.name}
                </h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
                    <div style="background: rgba(220,53,69,0.05); padding: 8px 10px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 16px; font-weight: 700; color: #DC3545;">${s.avg_temp} \u00B0C</div>
                        <div style="font-size: 9px; color: #8494A7; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Avg Temp</div>
                    </div>
                    <div style="background: rgba(59,130,246,0.05); padding: 8px 10px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 16px; font-weight: 700; color: #3B82F6;">${s.total_precip} mm</div>
                        <div style="font-size: 9px; color: #8494A7; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Total Precip</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
                    <div style="background: rgba(139,92,246,0.05); padding: 8px 10px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 16px; font-weight: 700; color: #8B5CF6;">${s.anomaly_days}</div>
                        <div style="font-size: 9px; color: #8494A7; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Anomaly Days</div>
                    </div>
                    <div style="background: ${color}0A; padding: 8px 10px; border-radius: 8px; text-align: center;">
                        <div style="font-size: 16px; font-weight: 700; color: ${color};">${s.risk_category}</div>
                        <div style="font-size: 9px; color: #8494A7; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;">Risk Level</div>
                    </div>
                </div>
                <div style="font-size: 10px; color: #8494A7; text-align: center; padding-top: 4px; border-top: 1px solid #EDF0F4;">
                    ${s.lat.toFixed(4)}\u00B0N, ${s.lon.toFixed(4)}\u00B0E
                </div>
            </div>
        `;

        marker.bindPopup(popupContent, {
            maxWidth: 300,
            className: 'climate-popup'
        });
    });
}
