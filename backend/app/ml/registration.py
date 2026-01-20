import cv2
import numpy as np

def register_images(img1_bytes, img2_bytes):
    """
    Aligns img2 (OP3) to match img1 (OP1) using SIFT features.
    
    CRITICAL FIX: Preserves color channels throughout the pipeline.
    - Loads images in COLOR mode
    - Converts to grayscale ONLY for SIFT feature detection
    - Applies homography to the COLOR image
    
    Returns:
        The registered (warped) version of img2 in BGR color format.
    """
    # 1. Decode in COLOR mode (BGR)
    img1_color = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2_color = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img1_color is None or img2_color is None:
        print("[ERROR] Failed to decode images in color mode")
        return None

    # 2. Optimized Processing for Large Maps
    # If imagery is massive (e.g. > 10MB or > 4000px), we downsample for SIFT speed/memory,
    # but apply the resulting homography to the full resolution or a high-res proxy.
    max_dim = 4000
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    
    scale1 = 1.0
    if max(h1, w1) > max_dim:
        scale1 = max_dim / float(max(h1, w1))
        img1_proc = cv2.resize(img1_color, (0,0), fx=scale1, fy=scale1)
    else:
        img1_proc = img1_color

    scale2 = 1.0
    if max(h2, w2) > max_dim:
        scale2 = max_dim / float(max(h2, w2))
        img2_proc = cv2.resize(img2_color, (0,0), fx=scale2, fy=scale2)
    else:
        img2_proc = img2_color

    # 3. Convert to grayscale for SIFT feature detection
    img1_gray = cv2.cvtColor(img1_proc, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2_proc, cv2.COLOR_BGR2GRAY)

    # 4. SIFT Detector (Enhanced Feature Sensitivity + Limit)
    sift = cv2.SIFT_create(nfeatures=8000) 
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        print("[ERROR] Failed to detect features in images")
        return None

    # 5. Match Features (FLANN based Matcher)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)

    # 6. Filter Good Matches (Lowe's Ratio Test)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print(f"[INFO] Found {len(good)} good matches")

    if len(good) > 15:
        # Rescale keypoints back to original img1/img2 coordinates if we resized
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale1
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / scale2

        # 7. Find Homography (RANSAC)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None:
            print("[ERROR] Failed to compute homography matrix")
            return None
        
        # 8. Warp COLOR img2 to match original img1 dimensions
        # We warp the ORIGINAL color image to avoid quality loss
        warped_img2_color = cv2.warpPerspective(img2_color, M, (w1, h1))
        
        print(f"[SUCCESS] Registration complete. Final Shape: {warped_img2_color.shape}")
        return warped_img2_color
    else:
        print(f"[WARNING] Insufficient matches ({len(good)}/15). Using raw alignment.")
        return None
