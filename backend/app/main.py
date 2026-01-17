from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.ml.pit_detector import detect_pits
from app.ml.classifier import analyze_survival_at_pits
from app.ml.registration import register_images
import cv2
import numpy as np
import io
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local dev)
load_dotenv()

# ==========================================
# EcoDrone AI Backend - "The Brain"
# Built by Soumoditya Das for Kshitij 2026
# ==========================================

app = FastAPI(
    title="EcoDrone AI API", 
    description="High-Performance Afforestation Monitoring System", 
    version="1.0.0",
    root_path="/api" # CRITICAL: Forces FastAPI to ignore the /api prefix in Vercel routes
)

# Enable CORS (Cross-Origin Resource Sharing)
# This allows our React Frontend (running on a different port) to talk to this Backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to the Vercel domain!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {
        "message": "EcoDrone AI Systems Online. Ready for Analysis.",
        "author": "Soumoditya Das",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Vercel and monitoring"""
    import cv2
    return {
        "status": "healthy",
        "opencv_version": cv2.__version__,
        "api_version": "1.0.0"
    }

@app.post("/analyze")
async def analyze_patch(
    op1_image: UploadFile = File(...),
    op3_image: UploadFile = File(...),
    model_type: str = "gemini" # Options: "gemini", "fast"
):
    import time
    start_time = time.time()
    
    try:
        print(f"\n{'='*60}")
        print(f"[REQUEST] Processing: {op1_image.filename} + {op3_image.filename}")
        print(f"[CONFIG] Model: {model_type.upper()}")
        print(f"{'='*60}")
        
        op1_bytes = await op1_image.read()
        op3_bytes = await op3_image.read()
        
        print(f"[INFO] OP1 size: {len(op1_bytes)} bytes")
        print(f"[INFO] OP3 size: {len(op3_bytes)} bytes")
        
        # 1. Detect Pits (OP1)
        print("\n[STEP 1] Detecting pits in OP1...")
        pits = detect_pits(op1_bytes)
        
        if not pits:
            print("[WARNING] No pits detected in OP1")
            # Try to proceed anyway if we can detecting pits in OP3 directly (classifier handles this)
            # But usually we need OP1 for the "Plan". 
            # For now, let's allow it to flow through, maybe classifier finds them in OP3.

        print(f"[SUCCESS] Detected {len(pits)} pits (Initial)")

        # 2. Register Images
        print("\n[STEP 2] Registering OP3 to OP1...")
        registered_op3 = register_images(op1_bytes, op3_bytes)
        
        registration_status = "success" if registered_op3 is not None else "gps_fallback"
        image_to_analyze = registered_op3 if registered_op3 is not None else op3_bytes
        
        if registration_status == "success":
            print("[SUCCESS] Images aligned successfully")
        else:
            print("[WARNING] Registration failed, using raw OP3 image")

        # 3. Analyze Survival
        print(f"\n[STEP 3] Analyzing survival with {model_type} model...")
        
        use_gemini = (model_type == "gemini")
        
        survival_stats = analyze_survival_at_pits(
            image_to_analyze, 
            pits, 
            use_gemini=use_gemini
        )
        
        if "error" in survival_stats:
            raise HTTPException(status_code=500, detail=survival_stats["error"])
        
        exec_time = round(time.time() - start_time, 2)
        print(f"\n[COMPLETE] Total Processing Time: {exec_time}s")
        print(f"[RESULTS] Survival Rate: {survival_stats.get('rate', 0):.1f}%")
        print(f"{'='*60}\n")
        
        # ... (rest of the response object construction) ...

        response = {
            "status": "success",
            "metrics": {
                "processing_time_sec": exec_time,
                "registration": registration_status,
                "total_pits": survival_stats.get('total', 0),
                "survival_rate": round(survival_stats.get('rate', 0), 2),
                "dead_count": survival_stats.get('dead', 0),
                "model_used": model_type
            },
            "casualties": [
                {"id": p.get('id', i), "x": p['x'], "y": p['y'], "conf": p['confidence'], "reason": p.get('reason', '')} 
                for i, p in enumerate(survival_stats.get('details', [])) 
                if p['status'] == 'dead'
            ],
            # We also send 'live' points for visualization if needed, or just full details
            "raw_details": survival_stats.get("details", [])
        }
        
        return response
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n[ERROR] Processing failed:")
        print(error_trace)
        raise HTTPException(
            status_code=500, 
            detail=f"Error: {str(e)}"
        )
@app.post("/report")
async def generate_report(data: dict):
    """
    Generates a professional field report based on analysis metrics.
    """
    import google.generativeai as genai
    import os
    
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return {"report": "Gemini API Key missing. Cannot generate AI report."}
        
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    metrics = data.get("metrics", {})
    dead_count = metrics.get("dead_count", 0)
    total = metrics.get("total_pits", 0)
    rate = metrics.get("survival_rate", 0)
    
    prompt = f"""
    You are an expert Forester and Data Scientist writing a field report for the Odisha Forest Department.
    Data:
    - Total Saplings Planted/Audited: {total}
    - Survival Rate: {rate}%
    - Dead Saplings Detected: {dead_count}
    - Location: Benkmura/Debadihi VF (Odisha)
    
    Task:
    Write a concise, professional, and "powerful" executive summary (max 150 words).
    - Analyze the survival rate (Good/Bad?).
    - Suggest 2 actionable steps for ground staff based on the dead count.
    - Use professional tone.
    - Format with Bullet points.
    """
    
    try:
        response = model.generate_content(prompt)
        return {"report": response.text}
    except Exception as e:
        return {"report": f"AI Generation Failed: {str(e)}"}
