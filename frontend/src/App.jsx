// ==========================================
// EcoDrone AI Frontend
// Built by Soumoditya Das for Kshitij 2026
// ==========================================
// This component handles the main layout, navigation, 
// and the "Deep Research" modal integration.

import { useState } from 'react';
import { motion } from 'framer-motion';
import { Leaf, Drone, Activity, Layers } from 'lucide-react';
import Dashboard from './components/Dashboard';
import './App.css';

function App() {
  const [showResearch, setShowResearch] = useState(false);
  const [modelType, setModelType] = useState('gemini');

  return (
    <div className="app-container">
      {/* Navigation / Header */}
      <nav className="glass-panel" style={{
        position: 'fixed', top: 20, left: '5%', right: '5%',
        zIndex: 1000, padding: '15px 30px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Drone color="var(--primary)" size={32} />
          <h2 style={{ fontSize: '1.5rem' }}>EcoDrone <span className="text-gradient">AI</span></h2>
        </div>

        <div style={{ display: 'flex', gap: '20px', alignItems: 'center' }}>
          {/* Model Selector */}
          <select
            className="glass-input"
            style={{ padding: '8px 15px', borderRadius: '8px', cursor: 'pointer' }}
            onChange={(e) => setModelType(e.target.value)}
          >
            <option value="gemini">✨ Gemini 1.5 Pro (Precision)</option>
            <option value="fast">⚡ Standard Fast (Heuristic)</option>
          </select>

          <span style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Built by Soumoditya Das | soumoditt@gmail.com
          </span>
          <button
            className="btn-secondary"
            onClick={() => setShowResearch(true)}
          >
            <Layers size={18} style={{ marginRight: 8, verticalAlign: 'text-bottom' }} />
            Deep Research
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main style={{ padding: '100px 5% 40px' }}>
        <Dashboard modelType={modelType} />
      </main>

      {/* Research Modal */}
      {showResearch && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.8)', zIndex: 2000,
          display: 'flex', justifyContent: 'center', alignItems: 'center'
        }} onClick={() => setShowResearch(false)}>
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel"
            style={{ width: '800px', maxHeight: '80vh', overflowY: 'auto', padding: '40px', background: '#0a1510' }}
            onClick={e => e.stopPropagation()}
          >
            <h2 className="text-gradient">Methodology & Research</h2>
            <br />
            <h3>1. Pit Detection (OP1)</h3>
            <p className="text-muted">
              We utilize OpenCV's Hough Circle Transform to identify 45cm diameter pits from 70m drone elevation (approx 2.5cm/px GSD).
              Algorithm logic accounts for soil texture noise using median blurring, aiming for the high precision standards of afforestation auditing.
            </p>

            <h3>2. Image Registration (Temporal Alignment)</h3>
            <p className="text-muted">
              To account for GPS drift (±1m), we employ SIFT (Scale-Invariant Feature Transform) feature matching and RANSAC homography estimation directly aligning OP3 orthomosaics to OP1 coordinates.
            </p>

            <h3>3. Sapling Classification & Bio-Alignment</h3>
            <p className="text-muted">
              Our approach is inspired by the <strong style={{ color: 'var(--primary)' }}>DeadTrees.Earth</strong> database initiative (University of Freiburg, Chair of Sensor-based Geoinformatics).
              Similar to Schiefer et al. (2023), we utilize spectral indices (Excess Green Index: 2G-R-B) fused with structural data to classify sapling vitality.
              Our pipeline supports plug-and-play YOLOv8 inference for scaling to large-scale forest mortality mapping.
            </p>

            <h3>4. Survival Calculation</h3>
            <code style={{ background: '#333', padding: '5px', borderRadius: '4px' }}>
              Survival % = (Total Pits - Dead Spots) / Total Pits
            </code>
            <br /><br />
            <div style={{ marginTop: '20px', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '10px' }}>
              <small className="text-muted">
                Reference: <a href="https://uni-freiburg.de/enr-geosense/research/deadtrees/" target="_blank" style={{ color: 'var(--secondary)' }}>DeadTrees.Earth Project</a>
              </small>
            </div>
            <br />
            <button className="btn-primary" onClick={() => setShowResearch(false)}>Close Research</button>
          </motion.div>
        </div>
      )}
    </div>
  );
}

export default App;
