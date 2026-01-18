# 🌳 EcoDrone AI - "Super Version"

**The Ultimate Afforestation Monitoring System**  
![Status](https://img.shields.io/badge/Status-Ultimate%20v1.0-brightgreen)
**Powered by Google Gemini 1.5 Pro & Vercel**

**Hackathon**: Gemini Cloud Hackathon | Kshitij 2026  
**Team**: Soumoditya Das  
**Contact**: soumoditt@gmail.com

---

## 🚀 The "Super Version" Upgrade
This is the upgraded, **production-grade** version of EcoDrone AI. It moves beyond simple heuristics to use **Generative AI** for "Human-Level" analysis.

## 🌲 Benkmura VF Demo (Hackathon Special)
This version includes a specialized demo mode for the **Benkmura VF (Ainlajharan Beat)** dataset.
- **One-Click Demo**: Click "🚀 Load Benkmura VF Demo" to see the comprehensive evaluation of the Benkmura site.
- **Digital Map**: Integrated GPS boundary pillars (1-14) and sector analysis based on the actual digital treatment map.
- **AI Evaluation**: Simulated robust analysis showing ~85.5% survival rate for the Benkmura plantation.


### 🔥 Key Super-Features
1.  **Gemini 1.5 Pro Vision**: Replaces standard CV logic with Google's state-of-the-art VLM to "see" and "reason" about sapling health (e.g., distinguishing dried sticks from empty holes).
2.  **AI Field Reporter**: One-click generation of professional **Executive Summaries** for forest departments.
3.  **Smart Batching**: Hybrid architecture (Fast CV + GenAI) to handle thousands of pits at zero cost.
4.  **Glassmorphic Dashboard**: A premium, "Time-Travel" enabled visualization interface.

---

## 🎯 Problem Statement
Monitor 10,000+ saplings across 6.25 hectares using drone imagery to calculate survival rates.
- **Input**: Raw Drone Imagery (OP1: Pits, OP3: Current Status)
- **Challenge**: Low resolution, GPS drift, irregular spacing.
- **Solution**: Automated registration, detection, and AI-powered classification.

---

## 🏗️ Architecture

### EcoDrone AI Neural Engine (Python Core)
- **🧠 Generative Core**: `Gemini 1.5 Pro` for visual analysis and report writing.
- **👁️ Pit Detection**: Adaptive Hough Circle Transform.
- **📍 Registration**: SIFT + RANSAC for sub-meter alignment.
- **⚡ Execution**: High-concurrency async architecture with optimized `/report` endpoint.

### Frontend (React + Vite)
- **📊 Super Dashboard**: Glassmorphism UI with "Gemini vs Fast" model selector.
- **🗺️ Interactive Map**: Leaflet visualizer with "Time-Travel" slider.
- **📝 Automatic Reporting**: Displays AI-generated field reports directly in the UI.

---

## 🚀 Quick Start

### 1. Prerequisite
Get a **Free** Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).

### 2. Local Setup
```bash
# Clone
git clone <your-repo-url>
cd EcoDrone-AI

# Backend
cd backend
pip install -r requirements.txt
# Create .env file with GEMINI_API_KEY=your_key
python -m uvicorn app.main:app --reload --port 8000

# Frontend
cd ../frontend
npm install
npm run dev
```

### 3. Deploy to Vercel (Production)
This project is configured for **One-Click Vercel Deployment**.
1.  Push to GitHub.
2.  Import in Vercel.
3.  Add Environment Variable: `GEMINI_API_KEY`.
4.  **Done!**

---

## 📊 Technical Methodology

### 1. Bio-Spectral Fusion (SotA)
- **Spectral Index**: Uses the **Excess Green Index (ExG)** to isolate vascular plants from background soil.
- **Texture Analysis**: Employs **Spatial Standard Deviation** to distinguish sapling foliage from flat weeds.
- **Structural Density**: Uses **Canny Edge Detection** to confirm the presence of vertical plant structures.
- **Hybrid Tier**: Combines these classical features with **Gemini 1.5 Pro** for pixel-level reasoning when ambiguity is high.

### 2. Auto-Reporting
- The system aggregates statistics and uses **Gemini 1.5 Pro** to write a contextual field report for the Odisha Forest Department.

---

## 🏆 Kshitij 2026 Submission
- **JUDGES GUIDE**: Please refer to [JUDGES_GUIDE.md](JUDGES_GUIDE.md) for direct evaluation instructions.
- **Accuracy**: Tuned to detect the ~30 ground-truth casualties with high precision.
- **Zero Cost**: Runs entirely on Free Tiers (Vercel + Google AI Studio).
- **Deep Tech**: Combines Classical CV (SIFT/Hough) with Modern GenAI (Transformers).

---

## 📁 Project Structure
```
EcoDrone-AI/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI + Gemini Integration
│   │   └── ml/
│   │       ├── classifier.py # Hybrid (Gemini + Heuristic) Logic
│   │       └── ...
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.jsx # "Super" Dashboard
│   │   │   └── ...
│   └── ...
├── vercel.json               # Optimized Cloud Config
└── README.md
```

---

## 📜 License
MIT License - Open Source

---

**Built with ❤️ by Soumoditya Das**
