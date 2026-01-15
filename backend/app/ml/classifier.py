import random
from PIL import Image
import torch
# from ultralytics import YOLO  # Uncomment for real YOLOv8

class SurvivalClassifier:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            # self.model = YOLO(model_path)
            pass

    def predict(self, image_patch):
        """
        Classifies a single image patch (PIL Image or numpy array) as Alive or Dead.
        Methodology inspired by DeadTrees.Earth (University of Freiburg).
        Returns:
            status: "alive" or "dead"
            confidence: float (0.0 to 1.0)
        """
        # --- REAL AI LOGIC PLACEHOLDER ---
        # if self.model:
        #     results = self.model(image_patch)
        #     # Parse YOLO result...
        #     return "alive", 0.95
        
        # --- ADVANCED HEURISTIC (ExG + Texture) ---
        # Methodology inspired by DeadTrees.Earth (University of Freiburg).
        # We look for green biomass (Index) and structural variation (Standard Deviation).
        
        if hasattr(image_patch, 'convert'):
             patch_np = np.array(image_patch.convert('RGB'))
        else:
             patch_np = image_patch
             
        if patch_np.size == 0 or patch_np.shape[0] < 2 or patch_np.shape[1] < 2: 
            return "dead", 0.0

        r, g, b = patch_np[:,:,0].astype(float), patch_np[:,:,1].astype(float), patch_np[:,:,2].astype(float)
        
        # Calculate Excess Green Index (ExG)
        # ExG = 2*G - R - B
        exg = 2.0 * g - r - b
        mean_exg = np.mean(exg)
        
        # Calculate Local Variation (to distinguish smooth soil from textured leaves)
        std_dev = np.std(patch_np)
        
        # Decision Logic:
        # 1. High Greenness (>25) -> High confidence Alive
        # 2. Moderate Greenness (15-25) + High Texture (>30) -> Likely Alive
        # 3. Low Greenness or Low Texture -> Labeled Dead/Empty Pit
        
        if mean_exg > 25: 
            return "alive", 0.92
        elif mean_exg > 15 and std_dev > 30:
            return "alive", 0.75
        else:
            # Low greenness or too smooth (just soil)
            conf = 1.0 - (mean_exg / 50.0)
            return "dead", min(max(conf, 0.5), 0.95)

def analyze_survival_at_pits(op3_image_input, pit_locations, gsd_cm_px=2.5):
    """
    Cropping patches at pit locations and running classification.
    Args:
        op3_image_input: bytes OR numpy array (BGR).
    """
    import numpy as np
    import cv2
    
    # Decode if bytes
    if isinstance(op3_image_input, bytes):
        nparr = np.frombuffer(op3_image_input, np.uint8)
        image_op3 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        image_op3 = op3_image_input
    
    if image_op3 is None:
        return {"error": "Could not decode OP3 image"}

    classifier = SurvivalClassifier()
    results = []
    
    # 1m diameter crop = 100cm / 2.5 = 40 pixels
    crop_size = int(100 / gsd_cm_px)
    half_crop = crop_size // 2
    
    h, w, _ = image_op3.shape
    
    # --- ADVANCED PIT DETECTION (Hough Circle Transform) ---
    # This section replaces the reliance on pre-defined pit_locations
    # and instead detects them dynamically.
    
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image_op3, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian blur to reduce noise and help HoughCircles
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # 3. Define expected circle radius range based on GSD
    # Assuming a pit diameter of ~10-20cm for saplings
    min_radius_cm = 10 # Smallest expected sapling/pit
    max_radius_cm = 20 # Largest expected sapling/pit
    
    min_radius = int(min_radius_cm / gsd_cm_px / 2) # Convert diameter to radius in pixels
    max_radius = int(max_radius_cm / gsd_cm_px / 2)
    
    # Ensure min_radius is at least 1 to avoid issues
    min_radius = max(1, min_radius)
    
    # 4. Hough Circle Transform with Adaptive Fallback
    # -----------------------------------------------------------------
    min_dist = int(250 / gsd_cm_px * 0.7) # Slightly stricter spacing
    
    circles = cv2.HoughCircles(
        gray_blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=min_dist,
        param1=50, 
        param2=30, 
        minRadius=min_radius, 
        maxRadius=max_radius
    )

    # Fallback: if no circles are found, try with a lower threshold
    if circles is None:
        circles = cv2.HoughCircles(
            gray_blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.5, 
            minDist=min_dist,
            param1=40, 
            param2=20, 
            minRadius=min_radius, 
            maxRadius=max_radius
        )

    detected_pits = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Ensure we don't return thousands of false positives (limit to reasonable density)
        # 10,000 saplings / 6.25 ha = ~1600 per ha. 
        # For a single image patch, we cap at a reasonable number.
        for (x, y, r) in circles[:2000]: 
            detected_pits.append({"x": int(x), "y": int(y), "r": int(r)})
            
    # Use detected_pits for classification
    pit_locations_to_process = detected_pits if detected_pits else pit_locations
    
    dead_count = 0
    
    for pit in pit_locations_to_process:
        cx, cy = pit['x'], pit['y']
        
        # Boundary check
        x1 = max(0, cx - half_crop)
        y1 = max(0, cy - half_crop)
        x2 = min(w, cx + half_crop)
        y2 = min(h, cy + half_crop)
        
        patch = image_op3[y1:y2, x1:x2]
        
        status, conf = classifier.predict(patch)
        
        result_entry = {
            "x": cx, "y": cy, 
            "status": status, 
            "confidence": conf
        }
        
        if status == "dead":
            dead_count += 1
            
        results.append(result_entry)
        
    survival_rate = ((len(pit_locations_to_process) - dead_count) / len(pit_locations_to_process)) * 100 if pit_locations_to_process else 0
    
    return {
        "rate": survival_rate,
        "total": len(pit_locations_to_process),
        "dead": dead_count,
        "details": results
    }
