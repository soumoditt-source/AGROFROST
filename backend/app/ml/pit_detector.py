import cv2
import numpy as np

def detect_pits(image_bytes: bytes, gsd_cm_px: float = 2.5):
    """
    Detects planting pits using Multi-Scale Hough Circle Transform.
    Optimized for high-res drone imagery with background soil noise.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Multi-Scale Pre-processing
    # Bilateral filter preserves edges while smoothing soil texture noise
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    target_radius = int(22.5 / gsd_cm_px)
    
    all_circles = []
    
    # Scale search (±15% altitude variation)
    for scale in [0.9, 1.0, 1.1]:
        r_min = int(target_radius * 0.7 * scale)
        r_max = int(target_radius * 1.3 * scale)
        min_dist = int(250 / gsd_cm_px * 0.7 * scale)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1.2, 
            minDist=min_dist,
            param1=50, # Canny threshold
            param2=35, # Accumulator threshold (Sensitivity)
            minRadius=r_min, 
            maxRadius=r_max
        )
        
        if circles is not None:
            all_circles.append(circles[0])

    if not all_circles:
        return []

    # Combine and Deduplicate (Non-Maximum Suppression logic)
    consolidated_circles = np.vstack(all_circles)
    
    detected_pits = []
    # Simple deduplication: if centers are within minDist, take the strongest one
    # For a PoC, we'll take the first unique set.
    visited = set()
    for (x, y, r) in consolidated_circles:
        grid_pos = (int(x // 10), int(y // 10))
        if grid_pos not in visited:
            detected_pits.append({"x": int(x), "y": int(y), "r": int(r)})
            visited.add(grid_pos)
            
    return detected_pits
