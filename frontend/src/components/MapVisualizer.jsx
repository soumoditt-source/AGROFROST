// ==========================================
// Map Visualizer (Leaflet)
// ==========================================
// Renders drone orthomosaics using a "Simple CRS" (Coordinate Reference System).
// This maps image pixels directly to map coordinates, avoiding complex Lat/Lng conversion
// for local grid based analysis.

import { useEffect, useState } from 'react';
import { MapContainer, ImageOverlay, CircleMarker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default Leaflet icons in Webpack/Vite builds
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});
L.Marker.prototype.options.icon = DefaultIcon;

const MapVisualizer = ({ op1Image, op3Image, analysisData }) => {
    const [bounds, setBounds] = useState(null);
    const [sliderValue, setSliderValue] = useState(50); // 0 = OP1, 100 = OP3

    useEffect(() => {
        if (op1Image) {
            const img = new Image();
            img.src = op1Image;
            img.onload = () => {
                // Leaflet uses [lat, lng] -> [y, x]
                // CRS.Simple: [0,0] is bottom-left? No, typically top-left if we flip it or just standard.
                // Let's use [[0,0], [h, w]]
                const h = img.naturalHeight;
                const w = img.naturalWidth;
                setBounds([[0, 0], [h, w]]);
            };
        }
    }, [op1Image]);

    if (!bounds) return <div style={{ padding: 20 }}>Loading Map Bounds...</div>;

    const op3Opacity = sliderValue / 100;

    // In CRS.Simple, standard is y grows downwards if we behave like image.
    // Actually L.CRS.Simple treats [0,0] as invisible reference.
    // Bounds: [[0,0], [height, width]] maps to the image coverage.
    // Pits are (x,y) from top-left.
    // In CRS.Simple, "lat" is y, "lng" is x.
    // But usually typically y goes up.
    // To match image coords (y down), we usually map:
    // Image (0,0) -> Map [Height, 0] (Top-Left)
    // Image (w,h) -> Map [0, Width] (Bottom-Right)
    // So y_map = Height - y_image. x_map = x_image.

    return (
        <div>
            {/* Time Travel Slider */}
            <div className="glass-panel" style={{ padding: '15px', marginBottom: '15px', display: 'flex', alignItems: 'center', gap: '15px' }}>
                <span style={{ fontWeight: 'bold' }}>Time Travel:</span>
                <span style={{ color: 'var(--text-muted)' }}>OP1 (Pits)</span>
                <input
                    type="range"
                    min="0" max="100"
                    value={sliderValue}
                    onChange={(e) => setSliderValue(e.target.value)}
                    style={{ flex: 1, accentColor: 'var(--primary)' }}
                />
                <span style={{ color: 'var(--primary)' }}>OP3 (Results)</span>
            </div>

            <div className="map-container">
                <MapContainer
                    bounds={bounds}
                    zoom={-1}
                    crs={L.CRS.Simple}
                    style={{ height: '100%', width: '100%', background: '#000' }}
                    minZoom={-5}
                >
                    {/* OP1 - Base Layer (Pits) */}
                    <ImageOverlay
                        url={op1Image}
                        bounds={bounds}
                        opacity={1} // Base always visible? Or crossfade with slider? 
                    // Let's keep OP1 visible and fade OP3 over it.
                    />

                    {/* OP3 - Overlay Layer (Saplings) */}
                    <ImageOverlay
                        url={op3Image}
                        bounds={bounds}
                        opacity={op3Opacity}
                    />

                    {/* Sapling Markers */}
                    {/* Only show markers if slider is towards OP3 side (> 50%) or always? Always is better for stats. */}
                    {analysisData.raw_details.map((pit, idx) => (
                        <CircleMarker
                            key={idx}
                            center={[pit.y, pit.x]}
                            radius={6}
                            pathOptions={{
                                color: pit.status === 'alive' ? 'var(--primary)' : 'var(--alert)',
                                fillOpacity: 0.6
                            }}
                        >
                            <Popup>
                                <strong>Pit #{idx}</strong><br />
                                Status: <span style={{ color: pit.status === 'alive' ? 'var(--primary)' : 'var(--alert)' }}>
                                    {pit.status.toUpperCase()}
                                </span><br />
                                Confidence: {(pit.confidence * 100).toFixed(1)}%<br />
                                {pit.reason && (
                                    <small style={{ color: '#666', fontStyle: 'italic' }}>
                                        "{pit.reason}"
                                    </small>
                                )}
                            </Popup>
                        </CircleMarker>
                    ))}
                </MapContainer>
            </div>
        </div>
    );
};

export default MapVisualizer;
