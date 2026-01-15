import cv2
import numpy as np

def register_images(img1_bytes, img2_bytes):
    """
    Aligns img2 (OP3) to match img1 (OP1) using SIFT features.
    
    Returns:
        The registered (warped) version of img2.
    """
    # 1. Decode
    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return None

    # 2. SIFT Detector (Enhanced Feature Sensitivity)
    sift = cv2.SIFT_create(nfeatures=5000) # Increased for high-res drone maps
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. Match Features (FLANN based Matcher)
    # ---------------------------------------
    # We use a Fast Library for Approximate Nearest Neighbors to find
    # similar keypoints between Year 1 (OP1) and Year X (OP3).
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # Higher checks = more precision, slower
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(des1, des2, k=2)

    # 4. Filter Good Matches (Lowe's Ratio Test)
    # ------------------------------------------
    # Discard ambiguous matches. If the best match is not significantly
    # better than the 2nd best, it's noise.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 10:
        # We need at least 4 points for Homography, but 10 is safe.
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 5. Find Homography (RANSAC)
        # ---------------------------
        # Calculate the perspective transformation matrix that maps OP3 onto OP1.
        # RANSAC ignores outliers (bad matches).
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # 6. Warp img2 to match img1
        # Apply the matrix to align the images perfectly.
        h, w = img1.shape
        warped_img2 = cv2.warpPerspective(img2, M, (w, h))
        return warped_img2
    else:
        print(f"Not enough matches are found - {len(good)}/10")
        return None
