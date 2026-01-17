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

const Dashboard = (props) => {
    const [op1File, setOp1File] = useState(null);
    const [op3File, setOp3File] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);

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
        setResult(null); // Clear previous results
        const formData = new FormData();
        formData.append('op1_image', op1File);
        formData.append('op3_image', op3File);
        formData.append('model_type', props.modelType || 'gemini');

        try {
            // Production-Ready: Using relative path for Vercel/Cloud deployment
            const response = await axios.post('/api/analyze', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
                timeout: 120000 // Extended timeout for GenAI (2 mins)
            });

            if (response.data.status === 'partial_error') {
                alert(`Warning: ${response.data.message}`);
            }

            setResult(response.data);
        } catch (error) {
            console.error("Analysis failed", error);

            let errorMessage = "Analysis failed! ";
            if (error.response) {
                // Server responded with error
                errorMessage += error.response.data.detail || error.response.statusText;
            } else if (error.request) {
                // Request made but no response
                errorMessage += "No response from server. Check your connection.";
            } else {
                // Something else happened
                errorMessage += error.message;
            }

            alert(errorMessage);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="dashboard">
            <motion.h1
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                style={{ textAlign: 'center', marginBottom: '40px' }}
            >
                Afforestation <span className="text-gradient">Monitor</span>
            </motion.h1>

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
                <div className="loader-container">
                    <div className="loading-spinner"></div>
                    <p style={{ marginTop: '20px', color: 'var(--primary)' }}>
                        {props.modelType === 'gemini' ? "Sending crops to Google Gemini Vision..." : "Aligning Images & Detecting Saplings..."}
                    </p>
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
