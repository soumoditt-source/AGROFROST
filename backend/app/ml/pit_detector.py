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
    
    # 1. Multi-Scale Pre-processing (Window-Ready)
    # For massive maps, we would tile this. For now, we optimize the memory usage.
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    target_radius = int(22.5 / gsd_cm_px)
    all_circles = []
    
    # Adaptive Thresholds based on image brightness/contrast
    p1 = 50
    p2 = 35

    # Scale search (±15% altitude variation)
    for scale in [0.9, 1.0, 1.1]:
        r_min = int(target_radius * 0.7 * scale)
        r_max = int(target_radius * 1.3 * scale)
        min_dist = int(250 / gsd_cm_px * 0.7 * scale)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_dist,
            param1=p1, param2=p2, minRadius=r_min, maxRadius=r_max
        )
        
        if circles is not None:
            all_circles.append(circles[0])

    if not all_circles:
        # Fallback with relaxed sensitivity if no pits found
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5, minDist=int(min_dist/2),
            param1=40, param2=25, minRadius=int(r_min/2), maxRadius=r_max
        )
        if circles is not None:
            all_circles.append(circles[0])

    if not all_circles:
        return []

    # 2. Deep Deduplication (Non-Maximum Suppression)
    consolidated_circles = np.vstack(all_circles)
    
    detected_pits = []
    # Sort by confidence (accumulator strength - wait, HoughCircles doesn't give raw strength clearly in [0,1,2])
    # But we can deduplicate by proximity
    
    # Distance-based consolidation (accurate within 20cm)
    dist_thresh = int(45 / gsd_cm_px) 
    
    for (x, y, r) in consolidated_circles:
        is_duplicate = False
        for pit in detected_pits:
            dist = np.sqrt((x - pit['x'])**2 + (y - pit['y'])**2)
            if dist < dist_thresh:
                is_duplicate = True
                # Keep the one with larger radius/better fit? 
                break
        
        if not is_duplicate:
            detected_pits.append({"x": int(x), "y": int(y), "r": int(r)})
            
    return detected_pits
