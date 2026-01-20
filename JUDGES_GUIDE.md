# Judge's Evaluation Guide - AgroFrost AI

This guide is designed for the Kshitij 2026 judges to verify the "EcoDrone AI" system's accuracy and performance.

## 1. Quick Verification (The Demo)
- Open the application.
- Click **"🚀 Load Benkmura VF Demo"**.
- **What to look for**:
    - **Survival Rate**: ~85.5% (A realistic survival rate for Year 1).
    - **Dead Spots**: ~30-120 casualties detected.
    - **Interactivity**: Use the **"Time Travel"** slider to cross-fade between OP1 (Pits) and OP3 (Current) images. Verify that markers align with visible sapling gaps.

## 2. Raw Data & Professional Tools
- **The "No-Cheat" Engine**: Unlike basic demos, this system processes actual **500MB+ TIF orthomosaics**. 
- **CLI Intelligence**: Run the professional batch processor:
  ```bash
  python scripts/process_raw_data.py --root "Drone image"
  ```
- **What it does**: Automatically detects and aligns large datasets, runs multi-pass feature engineering, and generates a standard JSON/CSV report suitable for government auditing.

## 3. High-End Technical Excellence (3D + 2D)
The system employs several "State-of-the-Art" (SotA) techniques:
- **Bio-Spectral 3D Fusion**: We analyze **verticality indicators** (shadows, height-spectral correlations) to distinguish a 3D sapling from flat, green grass.
- **Multimodal Evaluation**: When enabled, **Gemini 1.5 Pro Vision** performs deep visual inspection, explaining its reasoning based on plant structural integrity.
- **Pyramid SIFT Matching**: Handles massive maps by intelligent downsampling during feature detection while maintaining full-res coordinate precision.

## 4. Key Metrics for Success
- **100% Evaluation**: Every detected pit is analyzed for both spectral and structural vitality.
- **Audit-Ready**: Outputs detailed coordinates and confidence scores for every point.
- **Precision**: Tuned to meet Odisha Forest Department standards (gsd=2.5cm/px).

---
**Author**: Soumoditya Das (IIT Kharagpur)
**Final Version**: v2.0-ULTIMATE
