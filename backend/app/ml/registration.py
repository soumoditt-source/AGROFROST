from PIL import Image
import io
import cv2
import numpy as np
import os
import gc

# Allow processing of large drone orthomosaics
Image.MAX_IMAGE_PIXELS = None

def register_images(img1_bytes, img2_bytes):
    """
    Aligns img2 (OP3) to match img1 (OP1) using SIFT features.
    Memory-optimized for high-res TIF files.
    """
    # 1. Memory-Efficient Loading (Pillow)
    pil1 = Image.open(io.BytesIO(img1_bytes))
    pil2 = Image.open(io.BytesIO(img2_bytes))
    
    w1, h1 = pil1.size
    w2, h2 = pil2.size
    
    # Cap resolution for registration to save memory
    max_reg_dim = 1500
    
    scale1 = 1.0
    if max(w1, h1) > max_reg_dim:
        scale1 = max_reg_dim / float(max(w1, h1))
        proc1_pil = pil1.resize((int(w1 * scale1), int(h1 * scale1)), Image.BILINEAR)
    else:
        proc1_pil = pil1

    scale2 = 1.0
    if max(w2, h2) > max_reg_dim:
        scale2 = max_reg_dim / float(max(w2, h2))
        proc2_pil = pil2.resize((int(w2 * scale2), int(h2 * scale2)), Image.BILINEAR)
    else:
        proc2_pil = pil2

    # 2. Convert to OpenCV Format (Gray for SIFT)
    img1_gray = np.array(proc1_pil.convert('L'))
    img2_gray = np.array(proc2_pil.convert('L'))
    
    # Release processing pilllows
    if proc1_pil != pil1: del proc1_pil
    if proc2_pil != pil2: del proc2_pil

    # 3. SIFT Detector
    sift = cv2.SIFT_create(nfeatures=4000) # Reduced features for memory
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    del img1_gray
    del img2_gray
    gc.collect()

    # 4. Matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    good = []
    if matches:
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) > 10:
        # Rescale keypoints back to original img1/img2 coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2) / scale1
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2) / scale2

        # Find Homography (RANSAC)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        if M is None: return None
        
        # Load Color Images at Registration Resolution (Memory-safe)
        temp1 = pil1.resize((int(w1 * scale1), int(h1 * scale1)), Image.LANCZOS)
        img1_color = np.array(temp1)[:, :, ::-1].copy()
        del temp1
        del pil1
        
        temp2 = pil2.resize((int(w2 * scale2), int(h2 * scale2)), Image.LANCZOS)
        img2_color = np.array(temp2)[:, :, ::-1].copy()
        del temp2
        del pil2
        
        gc.collect()
        
        # Adjust M for the scaled space
        S1 = np.diag([scale1, scale1, 1])
        S2_inv = np.diag([1/scale2, 1/scale2, 1])
        M_scaled = S1 @ M @ S2_inv
        
        warped_img3 = cv2.warpPerspective(img2_color, M_scaled, (int(w1 * scale1), int(h1 * scale1)))
        
        return warped_img3
    else:
        # Fallback: Just return downsampled color image if registration fails
        temp2 = pil2.resize((int(w2 * scale2), int(h2 * scale2)), Image.LANCZOS)
        img2_color = np.array(temp2)[:, :, ::-1].copy()
        return img2_color
