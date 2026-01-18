// ==========================================
// Dashboard Component
// ==========================================
// Manages the user flow: 
// 1. Upload OP1/OP3 images
// 2. Send to Backend API
// 3. Display Survival Stats & Map
//
// Built for Kshitij 2026

import { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';
import { Upload, Cpu, Activity, AlertTriangle } from 'lucide-react';
import MapVisualizer from './MapVisualizer';
import { benkmuraData } from '../data/benkmura_data';

const Dashboard = (props) => {
    const [op1File, setOp1File] = useState(null);
    const [op3File, setOp3File] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [logs, setLogs] = useState([]);

    const addLog = (msg) => {
        setLogs(prev => [...prev.slice(-9), `[${new Date().toLocaleTimeString()}] ${msg}`]);
    };

    // Previews
    const [op1Preview, setOp1Preview] = useState(null);
    const [op3Preview, setOp3Preview] = useState(null);

    const handleFileChange = (e, type) => {
        const file = e.target.files[0];
        if (file) {
            // Validate file type
            if (!file.type.startsWith('image/')) {
                alert('Please upload a valid image file (PNG, JPG, etc.)');
                return;
            }

            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert('File size too large. Please upload images smaller than 10MB.');
                return;
            }

            if (type === 'op1') {
                setOp1File(file);
                setOp1Preview(URL.createObjectURL(file));
            } else {
                setOp3File(file);
                setOp3Preview(URL.createObjectURL(file));
            }
        }
    };

    const runAnalysis = async () => {
        if (!op1File || !op3File) {
            alert("Please upload both OP1 and OP3 images.");
            return;
        }

        setLoading(true);
        setResult(null);
        setLogs(["[SYSTEM] Initializing Neural Engine...", "[INFO] Preparing OP1/OP3 payloads..."]);

        const formData = new FormData();
        formData.append('op1_image', op1File);
        formData.append('op3_image', op3File);
        formData.append('model_type', props.modelType || 'gemini');

        try {
            addLog(`Uplinking to ${props.modelType === 'gemini' ? "Google Gemini 1.5 Pro" : "Heuristic Core"}...`);

            // Step-by-step logic simulation for UI logs
            setTimeout(() => addLog("Registering Temporal Orthomosaics (SIFT + RANSAC)..."), 3000);
            setTimeout(() => addLog("Detecting Pits in OP1 (Multi-Scale Hough)..."), 6000);
            setTimeout(() => addLog("Analyzing Survival with Bio-Spectral Fusion..."), 9000);

            const response = await axios.post('/api/analyze', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000
            });

            addLog("Analysis Complete. Processing Results...");
            setResult(response.data);
            addLog(`Success: Detected ${response.data.metrics.total_pits} pits with ${response.data.metrics.survival_rate}% survival.`);

        } catch (error) {
            addLog(`[ERROR] ${error.message}`);
            // ... (rest of error handling) ...
        } finally {
            setLoading(false);
        }
    };

    const loadDemo = () => {
        setLoading(true);
        // Simulate loading time
        setTimeout(() => {
            setOp1Preview('/assets/benkmura_op1.png');
            setOp3Preview('/assets/benkmura_op3.png');
            setOp1File(new File([""], "benkmura_op1.png")); // Dummy for validation
            setOp3File(new File([""], "benkmura_op3.png"));

            // Mock Result
            setResult({
                metrics: {
                    survival_rate: 85.5,
                    processing_time_sec: 2.1,
                    dead_count: 120
                },
                raw_details: benkmuraData.boundaryPillars.map((p, index) => {
                    // Deterministic "Real-World" simulation based on provided map data
                    // Simulate some clusters of "dead" saplings (e.g., near index 4 and 10)
                    const isDead = [4, 10, 11].includes(index);
                    return {
                        x: (p.lng - 83.818044) * 12000,
                        y: (p.lat - 21.652007) * 12000,
                        status: isDead ? 'dead' : 'alive',
                        confidence: isDead ? 0.88 : 0.96,
                        reason: isDead ? 'Dried foliage (Lack of moisture)' : null
                    };
                }),
                casualties: [],
                ai_report: `**Benkmura VF Analysis Report**\n\nProject: ${benkmuraData.projectName}\nLocation: ${benkmuraData.location}\nDetected robust growth in Sectors 1-4. \nSurvival Rate: 85.5%`
            });
            setLoading(false);
        }, 1500);
    };

    return (
        <div className="dashboard">
            <motion.h1
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ textAlign: 'center', marginBottom: '20px' }}
            >
                Afforestation <span className="text-gradient">Monitor</span> <span style={{ fontSize: '0.8rem', opacity: 0.6, marginLeft: '10px', padding: '2px 8px', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '12px' }}>v1.0</span>
            </motion.h1>

            <div style={{ textAlign: 'center', marginBottom: '30px' }}>
                <button onClick={loadDemo} className="btn-secondary" style={{ padding: '10px 20px' }}>
                    🚀 Load Benkmura VF Demo (Hackathon Mode)
                </button>
            </div>

            {/* Project Details (Only in Demo Mode) */}
            {result && result.ai_report && result.ai_report.includes('Benkmura') && (
                <div className="glass-panel" style={{ margin: '0 auto 30px', maxWidth: '800px', padding: '20px' }}>
                    <h3 style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '10px', marginBottom: '15px' }}>
                        🌲 Project Metadata: {benkmuraData.projectName}
                    </h3>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', fontSize: '0.9rem' }}>
                        <div>
                            <p><strong style={{ color: 'var(--primary)' }}>Location:</strong> {benkmuraData.location}</p>
                            <p><strong>Year:</strong> {benkmuraData.year}</p>
                            <p><strong>Model:</strong> {benkmuraData.modelType}</p>
                            <p><strong>Area:</strong> {benkmuraData.totalArea}</p>
                        </div>
                        <div>
                            <p><strong>Seedlings:</strong> {benkmuraData.totalSeedlings}</p>
                            <p><strong>Sectors:</strong> {benkmuraData.sectors.length}</p>
                            <p><strong>Species:</strong> {benkmuraData.sectors[0].species.slice(0, 3).join(', ')} + more</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Input Section */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
                <div className="glass-panel" style={{ padding: '30px', textAlign: 'center' }}>
                    <h3>1. Upload OP1 (Pits)</h3>
                    <p style={{ color: 'var(--text-muted)' }}>Drone imagery before planting</p>
                    <div style={{ margin: '20px 0', border: '2px dashed rgba(255,255,255,0.2)', borderRadius: '12px', padding: '20px' }}>
                        {op1Preview ? (
                            <img src={op1Preview} alt="OP1" style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px' }} />
                        ) : (
                            <Upload size={48} color="var(--text-muted)" />
                        )}
                    </div>
                    <input type="file" onChange={(e) => handleFileChange(e, 'op1')} style={{ color: '#fff' }} />
                </div>

                <div className="glass-panel" style={{ padding: '30px', textAlign: 'center' }}>
                    <h3>2. Upload OP3 (Current)</h3>
                    <p style={{ color: 'var(--text-muted)' }}>Latest drone imagery (Year 1/2/3)</p>
                    <div style={{ margin: '20px 0', border: '2px dashed rgba(255,255,255,0.2)', borderRadius: '12px', padding: '20px' }}>
                        {op3Preview ? (
                            <img src={op3Preview} alt="OP3" style={{ maxWidth: '100%', maxHeight: '200px', borderRadius: '8px' }} />
                        ) : (
                            <Upload size={48} color="var(--text-muted)" />
                        )}
                    </div>
                    <input type="file" onChange={(e) => handleFileChange(e, 'op3')} style={{ color: '#fff' }} />
                </div>
            </div>

            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
                <button className="btn-primary" onClick={runAnalysis} disabled={loading} style={{ fontSize: '1.2rem', padding: '15px 40px' }}>
                    {loading ? (
                        <span>
                            Processing with {props.modelType === 'gemini' ? "Gemini 1.5 Pro" : "Standard Algorithm"}...
                        </span>
                    ) : "Analyze Patch"} <Cpu size={20} style={{ marginLeft: 10 }} />
                </button>
            </div>

            {loading && (
                <div style={{ margin: '20px auto', maxWidth: '800px' }}>
                    <div className="glass-panel" style={{
                        background: '#000',
                        padding: '15px',
                        fontFamily: 'monospace',
                        color: 'var(--primary)',
                        border: '1px solid var(--primary)',
                        maxHeight: '200px',
                        overflowY: 'auto'
                    }}>
                        {logs.map((log, i) => <div key={i}>{log}</div>)}
                        <div className="loading-spinner" style={{ width: 14, height: 14, display: 'inline-block', marginLeft: 10 }}></div>
                    </div>
                </div>
            )}

            {/* Results Section */}
            {result && result.raw_details && result.raw_details.length > 0 && (
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="results-container"
                >
                    <div className="stats-grid">
                        <div className="glass-panel stat-card">
                            <Activity color="var(--primary)" size={32} />
                            <h3>{result.metrics.survival_rate.toFixed(1)}%</h3>
                            <p className="text-muted">Survival Rate</p>
                        </div>
                        <div className="glass-panel stat-card">
                            <Cpu color="var(--secondary)" size={32} />
                            <h3>{result.metrics.processing_time_sec}s</h3>
                            <p className="text-muted">Processing Time</p>
                        </div>
                        <div className="glass-panel stat-card">
                            <AlertTriangle color="var(--alert)" size={32} />
                            <h3>{result.metrics.dead_count}</h3>
                            <p className="text-muted">Detected Dead Spots</p>
                        </div>
                    </div>

                    <div style={{ textAlign: 'center', marginBottom: '30px', display: 'flex', justifyContent: 'center', gap: '20px' }}>
                        <button
                            className="btn-secondary"
                            onClick={() => {
                                const headers = "ID,X_Coordinate,Y_Coordinate,Confidence,Reason\n";
                                const csvContent = "data:text/csv;charset=utf-8,"
                                    + headers
                                    + result.casualties.map(c => `${c.id},${c.x},${c.y},${c.conf},"${c.reason || 'N/A'}"`).join("\n");
                                const encodedUri = encodeURI(csvContent);
                                const link = document.createElement("a");
                                link.setAttribute("href", encodedUri);
                                link.setAttribute("download", `ecodrone_casualties_${Date.now()}.csv`);
                                document.body.appendChild(link);
                                link.click();
                            }}
                        >
                            Download Casualty CSV
                        </button>

                        <button
                            className="btn-primary"
                            style={{ background: 'var(--secondary)', borderColor: 'var(--secondary)' }}
                            onClick={async () => {
                                try {
                                    alert("Generating AI Report... This may take a few seconds.");
                                    const res = await axios.post('/api/report', { metrics: result.metrics });
                                    setResult(prev => ({ ...prev, ai_report: res.data.report }));
                                } catch (e) {
                                    alert("Failed to generate report");
                                }
                            }}
                        >
                            📝 Generate AI Field Report
                        </button>
                    </div>

                    {result.ai_report && (
                        <div className="glass-panel" style={{ padding: '25px', marginBottom: '30px', borderLeft: '4px solid var(--secondary)' }}>
                            <h3 style={{ color: 'var(--secondary)', marginBottom: '15px' }}>🤖 Gemini Field Report</h3>
                            <div style={{ lineHeight: '1.6', whiteSpace: 'pre-line' }}>
                                {result.ai_report}
                            </div>
                        </div>
                    )}

                    <div className="glass-panel" style={{ padding: '20px' }}>
                        <h2 style={{ marginBottom: '20px' }}>Interactive Survival Map</h2>
                        <MapVisualizer
                            op1Image={op1Preview}
                            op3Image={op3Preview}
                            analysisData={result}
                        />
                    </div>
                </motion.div>
            )}
        </div>
    );
};

export default Dashboard;
