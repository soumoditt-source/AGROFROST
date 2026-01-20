import random
import numpy as np
import cv2
from PIL import Image
import os
import google.generativeai as genai
import time
import json
from dotenv import load_dotenv

# Load environment variables from .env file (for local dev)
load_dotenv()

# ==============================================================================
# GEMINI API CONFIGURATION
# ==============================================================================
# We prioritize the environment variable, but you can also hardcode for testing (not recommended for prod)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARNING] GEMINI_API_KEY not found in environment variables. Gemini features will fail.")

# ==============================================================================
# CLASSIFIER LOGIC
# ==============================================================================

class SurvivalClassifier:
    def __init__(self, mode="gemini"):
        self.mode = mode  # "gemini" or "heuristic"
        self.model = None
        if self.mode == "gemini" and GEMINI_API_KEY:
            # Using the latest Gemini 1.5 Pro (Vision) model
            self.model = genai.GenerativeModel('gemini-1.5-pro')

    def predict(self, image_patch):
        """
        Classifies a single image patch.
        Args:
            image_patch: numpy array (BGR) or PIL Image
        Returns:
            status: "alive" or "dead"
            confidence: float (0.0 to 1.0)
            reason: str (optional explanation)
        """
        # Convert to PIL Image if numpy array
        if isinstance(image_patch, np.ndarray):
             # CV2 is BGR, PIL needs RGB
             rgb_patch = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
             pil_image = Image.fromarray(rgb_patch)
        else:
             pil_image = image_patch

        # ---------------------------------------------------------
        # MODE 1: GOOGLE GEMINI 1.5 PRO VISION (The "Ultimate" Way)
        # ---------------------------------------------------------
        if self.mode == "gemini" and self.model:
            try:
                # Construct the prompt
                prompt = """
                Analyze this high-resolution drone image crop of a reforestation pit (approx 1m Ground Sampling Distance).
                
                DETERMINE if there is a LIVING sapling/tree in the center or if it is DEAD/EMPTY.
                
                REASONING (3D & 2D):
                - Look for green pigments (2D Spectral).
                - Identify 3D STRUCTURAL indicators: does the plant cast a shadow? Is there a clearly defined central stem or crown structure indicating height?
                - Distinguish from flat grass/weeds: trees have verticality and unique shadow patterns.
                - Dead signs: Dried sticks (white/brown), empty hole with uniform soil, or clearly withered foliage.
                
                Return a JSON object with:
                - "status": "alive" or "dead"
                - "confidence": float (0.0 to 1.0)
                - "3d_confidence": float (0.0 to 1.0) based on verticality/shadow indicators
                - "reason": professional evaluation (max 15 words)
                """
                
                # Call Gemini API
                # Note: For production with thousands of pits, you'd want to batch these or use asyncio.
                # For per-request analysis, this is acceptable for the Hackathon demo.
                response = self.model.generate_content([prompt, pil_image])
                
                # Parse Response (Handle potential markdown wrapping)
                text_response = response.text.strip()
                if text_response.startswith("```json"):
                    text_response = text_response.replace("```json", "").replace("```", "")
                
                result = json.loads(text_response)
                
                return result.get("status", "dead").lower(), result.get("confidence", 0.0), result.get("reason", "Gemini Analysis")
                
            except Exception as e:
                print(f"[ERROR] Gemini API failed: {e}. Falling back to heuristic.")
                # Fallback to heuristic if API fails (rate limits, network, etc.)
                return self._heuristic_analysis(pil_image)

        # ---------------------------------------------------------
        # MODE 2: HEURISTIC (ExG + Texture) - Fallback/Fast Mode
        # ---------------------------------------------------------
        return self._heuristic_analysis(pil_image)

    def _heuristic_analysis(self, pil_image):
        """
        Advanced Bio-Spectral Fusion (ExG + Texture + Structural Density).
        Optimized for sapling survival detection in Year 1-3.
        """
        patch_np = np.array(pil_image)
        if patch_np.size == 0: 
            return "dead", 0.0, "Empty Patch"

        # 1. Spectral Analysis (Excess Green Index)
        # Convert to float for calculation
        r, g, b = patch_np[:,:,0].astype(float), patch_np[:,:,1].astype(float), patch_np[:,:,2].astype(float)
        exg = 2.0 * g - r - b
        mean_exg = np.mean(exg)
        
        # 2. Texture Complexity (Spatial Variation)
        gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
        std_dev = np.std(gray)
        
        # 3. Structural Density (Edge Detection)
        # Healthy saplings have distinct leaf edges/shadows
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # Decision Matrix (Multi-Factor Fusion)
        if mean_exg > 22 and edge_density > 0.02:
            conf = min(0.95, 0.7 + (mean_exg / 100.0) + edge_density)
            return "alive", conf, "High Bio-Spectral Signature"
        elif mean_exg > 15 and std_dev > 35:
            return "alive", 0.82, "Texture-Dominant Growth"
        elif mean_exg < 5 and edge_density < 0.005:
            return "dead", 0.94, "Bare Soil Detected (Low Indices)"
        else:
            # Ambiguous case - look at ExG primarily
            if mean_exg > 10:
                return "alive", 0.65, "Weak Vegetation Signal"
            else:
                return "dead", 0.75, "High Probability Casualty"


def analyze_survival_at_pits(op3_image_input, pit_locations, gsd_cm_px=2.5, use_gemini=True):
    """
    Main entry point for batch analysis.
    """
    # ... Decode logic ...
    if isinstance(op3_image_input, bytes):
        nparr = np.frombuffer(op3_image_input, np.uint8)
        image_op3 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        image_op3 = op3_image_input
    
    if image_op3 is None:
        return {"error": "Could not decode OP3 image"}
    
    # Initialize Classifier
    mode = "gemini" if use_gemini and GEMINI_API_KEY else "heuristic"
    print(f"[INFO] Initializing Survival Classifier in [{mode.upper()}] mode")
    classifier = SurvivalClassifier(mode=mode)
    
    results = []
    crop_size = int(100 / gsd_cm_px) # 1m box
    half_crop = crop_size // 2
    
    # --- ADVANCED PIT DETECTION (Hough Circle Transform) ---
    detected_pits = _detect_circles(image_op3, gsd_cm_px)

    # Use detected_pits for classification, fallback to provided pit_locations
    pit_locations_to_process = detected_pits if detected_pits else pit_locations
    
    if not pit_locations_to_process:
        print("[WARNING] No pit locations available for analysis")
        return {"rate": 0, "total": 0, "dead": 0, "details": []}
    
    results, dead_count = _process_pits(classifier, mode, image_op3, pit_locations_to_process, half_crop)
    
    total_pits = len(pit_locations_to_process)
    survival_rate = ((total_pits - dead_count) / total_pits) * 100 if total_pits > 0 else 0
    
    return {
        "rate": survival_rate,
        "total": total_pits,
        "dead": dead_count,
        "details": results
    }

def _process_pits(classifier, mode, image, pits, half_crop):
    """
    Helper to process the batch of pits.
    """
    gemini_limit = 15 
    processed_with_gemini = 0
    dead_count = 0
    results = []
    
    total_pits = len(pits)
    print(f"[INFO] Analyzing {total_pits} pits...")
    
    # Detect coordinate scaling
    max_x = max([p['x'] for p in pits]) if pits else 0
    max_y = max([p['y'] for p in pits]) if pits else 0
    h, w, _ = image.shape
    
    scale_x = w / max_x if max_x > w else 1.0
    scale_y = h / max_y if max_y > h else 1.0
    
    # Use the minimum scale to be conservative
    scale = min(scale_x, scale_y)
    if scale < 0.95:
        print(f"[INFO] Auto-scaling coordinates by {scale:.4f} to match image resolution ({w}x{h})")
    
    current_half_crop = int(half_crop * scale)
    
    for i, pit in enumerate(pits):
        cx, cy = int(pit['x'] * scale), int(pit['y'] * scale)
        
        # Boundary checks
        x1, y1 = max(0, cx - current_half_crop), max(0, cy - current_half_crop)
        x2, y2 = min(w, cx + current_half_crop), min(h, cy + current_half_crop)
        
        patch = image[y1:y2, x1:x2]
        
        # Decide Strategy (Simple Rate limiting)
        use_gemini = (mode == "gemini" and processed_with_gemini < gemini_limit)
        if use_gemini:
            processed_with_gemini += 1
            if processed_with_gemini % 2 == 0: time.sleep(0.5)
        
        status, conf, reason = _get_prediction(classifier, patch, use_gemini)
        if mode == "gemini" and not use_gemini: reason += " (Fast Mode)"

        if status == "dead":
            dead_count += 1
            
        results.append({"id": i, "x": cx, "y": cy, "status": status, "confidence": round(conf, 2), "reason": reason})
        
        if i % 10 == 0: print(f"Processed {i}/{total_pits}...")
            
    return results, dead_count

def _get_prediction(classifier, patch, use_gemini):
    """Encapsulates prediction logic to reduce complexity."""
    if patch.size == 0:
        return "dead", 0.0, "Empty Patch"
        
    if use_gemini:
        return classifier.predict(patch)
    else:
        rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        return classifier._heuristic_analysis(Image.fromarray(rgb_patch))

def _detect_circles(image, gsd_cm_px):
    """Helper to detect circles using Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    min_radius_cm = 10 
    max_radius_cm = 20
    min_radius = max(1, int(min_radius_cm / gsd_cm_px / 2))
    max_radius = int(max_radius_cm / gsd_cm_px / 2)
    min_dist = int(250 / gsd_cm_px * 0.7)
    
    print(f"[INFO] Detecting circles with radius range: {min_radius}-{max_radius}px")
    
    circles = cv2.HoughCircles(
        gray_blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )

    if circles is None:
        print("[INFO] Relaxing detection parameters...")
        circles = cv2.HoughCircles(
            gray_blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=min_dist,
            param1=40, param2=20, minRadius=min_radius, maxRadius=max_radius
        )
    
    detected_pits = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles[:2000]: 
            detected_pits.append({"x": int(x), "y": int(y), "r": int(r)})
        print(f"[SUCCESS] Detected {len(detected_pits)} potential sapling locations")
    else:
        print("[WARNING] No circles detected")
        
    return detected_pits
