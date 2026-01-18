# 🧮 EcoDrone AI: Mathematical Framework

This document details the mathematical models and algorithms powering the "Ultimate Version" of EcoDrone AI. We combine **Classical Computer Vision** (deterministic, fast) with **Generative AI** (probabilistic, semantic) for optimal performance.

---

## 1. Excess Green Index (ExG)
**Purpose**: Rapidly segment vegetation from soil/background in the "Fast Mode" heuristic analysis.

### Formula:
$$ ExG = 2 \cdot G - R - B $$

Where $R, G, B$ are the normalized pixel values (0-1) from the drone imagery.
*   **Logic**: Vegetation reflects significantly more Green than Red or Blue. Soil typically has higher Red/Blue components.
*   **Thresholding**:
    *   If $ExG > T$ (where $T \approx 25$): Classified as **Vegetation (Alive)**.
    *   Otherwise: Classified as **Non-Vegetation (Dead/Empty)**.

---

## 2. Hough Circle Transform
**Purpose**: Automating the detection of pits (45x45cm) from the OP1 (Pre-Planting) drone imagery.

### Algorithm:
The pit boundary is modeled as a circle equation:
$$ (x - a)^2 + (y - b)^2 = r^2 $$
Where $(a, b)$ is the center and $r$ is the radius.

1.  **Edge Detection**: We apply Gaussian Blur followed by Canny Edge Detection to find gradients.
2.  **Voting Space**: Each edge point "votes" for potential center coordinates $(a, b)$ based on the gradient direction.
3.  **Accumulator**: Local maxima in the accumulator array correspond to the centers of the pits.

**Implementation**:
We use `cv2.HoughCircles` with dynamic parameter relaxation (Adaptive Thresholding) to handle varying lighting conditions in the forest patch.

---

## 3. Image Registration (Alignment)
**Purpose**: Aligning the OP1 (Pits) image with the OP3 (Current) image to find exact sapling locations even if GPS drift occurs.

### Method: SIFT + RANSAC
1.  **SIFT (Scale-Invariant Feature Transform)**:
    *   Detect "Keypoints" (corners, blobs) invariant to scale and rotation.
    *   Generate descriptors vectors for these keypoints.
2.  **Feature Matching**:
    *   Match descriptors between OP1 and OP3 using k-Nearest Neighbors (k-NN).
3.  **RANSAC (Random Sample Consensus)**:
    *   Mathematically estimate the **Homography Matrix ($H$)** that maps points from OP3 to OP1.
    *   $H$ is a $3 \times 3$ matrix:
    $$ \begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} $$
    *   RANSAC iteratively selects random subsets of matches to find the $H$ that fits the most "inliers", rejecting outliers (noise).

---

## 4. Generative AI (The "Brain")
**Purpose**: Handling edge cases where simple math fails (e.g., dry leaves vs. dead stick).

### Model: Gemini 1.5 Pro (Vision)
Instead of a mathematical formula, we use a **Probabilistic Transformer Model**.
*   **Input**: Tensor of pixel values $I \in \mathbb{R}^{H \times W \times 3}$.
*   **Process**: The Vision Transformer (ViT) encodes the image into latent embeddings. The LLM decoder attends to specific tokens ("leaves", "greenery", "dried") based on the prompt.
*   **Output**: Probability distribution over tokens: $P(\text{"alive"} | I, \text{prompt})$.

---

---

## 5. Forestry Density Models
EcoDrone AI is programmed to support the Odisha Forest Department's standard models:
*   **ANR 200/500**: Used in canopy densities of 40-70%.
*   **AR 1000**: Used in canopy densities of 10-40%.
*   **AR 1600 (Benkmura Case)**: Used in bald patches (0-10% canopy). 
    *   **Spacing Math**: At 1600 saplings/Ha, spacing is approx $2.5m \times 2.5m$.
    *   **Automation**: Our circle detector uses a `minDist` parameter calculated as $2.5m / GSD$ to prevent overlapping detections and ensure accurate AR 1600 auditing.

---

**Summary**:
*   ExG + Hough = **Speed** ($O(N)$ complexity).
*   Gemini 1.5 Pro = **Intelligence** (High-dimensional semantic understanding).
*   **AR 1600 Optimized**: Specifically tuned for the Benkmura VF dataset.
*   **Combined**: The "Ultimate" Architecture.

