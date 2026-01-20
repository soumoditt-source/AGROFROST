from PIL import Image
import io
import cv2
import numpy as np
import os

# Allow processing of large drone orthomosaics
Image.MAX_IMAGE_PIXELS = None

def detect_pits(image_bytes: bytes, gsd_cm_px: float = 2.5):
    """
    Detects planting pits using memory-efficient Contour Analysis.
    Optimized for massive 100MP+ TIF orthomosaics.
    """
    # 1. Memory-Efficient Loading (Pillow)
    img_pil = Image.open(io.BytesIO(image_bytes))
    orig_w, orig_h = img_pil.size
    
    max_dim = 2500
    scale = 1.0
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / float(max(orig_w, orig_h))
        img_pil = img_pil.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
    
    # Convert to OpenCV Format (Gray)
    image_np = np.array(img_pil.convert('L')) 
    gsd_proc = gsd_cm_px / scale
    
    # Release Pillow memory immediately
    del img_pil
    
    # 2. Multi-Scale Pre-processing (Memory-Efficient)
    blurred = cv2.GaussianBlur(image_np, (7, 7), 1.5)
    
    # Release raw np image
    del image_np
    
    # 3. Contour-Based Detection
    # Pits are dark circular regions in lighter soil
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_circles = []
    # Filtering criteria for pits: size and circularity
    min_area = (20 / gsd_proc)**2 * np.pi / 4 # 20cm diameter min
    max_area = (70 / gsd_proc)**2 * np.pi / 4 # 70cm diameter max
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            perl = cv2.arcLength(cnt, True)
            if perl > 0:
                circularity = 4 * np.pi * area / (perl * perl)
                if circularity > 0.6: # circular enough
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    all_circles.append((x, y, radius))
    
    # Release processing buffers
    del blurred
    del thresh
    
    if not all_circles:
        return []

    # 4. Global Deduplication
    detected_pits = []
    dist_thresh = int(45 / gsd_proc) 
    
    for (x, y, r) in all_circles:
        is_duplicate = False
        for pit in detected_pits:
            dist = np.sqrt((x - pit['x'])**2 + (y - pit['y'])**2)
            if dist < dist_thresh:
                is_duplicate = True
                break
        
        if not is_duplicate:
            # Scale coordinates back up to original full-res
            detected_pits.append({
                "x": int(x / scale), 
                "y": int(y / scale), 
                "r": int(r / scale)
            })
            
    return detected_pits
