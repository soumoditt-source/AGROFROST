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
        
        # --- HEURISTIC FALLBACK (Greenery Detection) ---
        # "Is it green? Then it's alive." 
        # We use the Excess Green Index (ExG) which is standard in agriculture 
        # for separating plants from soil/residue.
        # Formula: ExG = 2*Green - Red - Blue
        if hasattr(image_patch, 'convert'):
             patch_np = np.array(image_patch.convert('RGB'))
        else:
             patch_np = image_patch
             
        if patch_np.size == 0: return "dead", 0.0

        r, g, b = patch_np[:,:,0], patch_np[:,:,1], patch_np[:,:,2]
        
        # Calculate ExG for every pixel
        exg = 2.0 * g - r - b
        mean_exg = np.mean(exg)
        
        # If mean excess green is high, it's likely a health sapling.
        # Threshold (>20) is empirical based on typical soil contrast.
        if mean_exg > 20: 
            return "alive", 0.8  # Strong Photosynthesis signal
        else:
            return "dead", 0.6   # Low greenness (bare soil / dry grass)

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
    
    dead_count = 0
    
    for pit in pit_locations:
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
        
    survival_rate = ((len(pit_locations) - dead_count) / len(pit_locations)) * 100 if pit_locations else 0
    
    return {
        "rate": survival_rate,
        "total": len(pit_locations),
        "dead": dead_count,
        "details": results
    }
