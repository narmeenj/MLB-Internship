# Day 15 Mini-Project: Document Image Enhancement Tool

An interactive computer vision preprocessing utility created with Python, OpenCV, and Gradio to clean up scan documents, level out geometry artifacts, and optimize text contrast.

## 🛠️ Implemented Architectural Concepts

### 1. Image Transformations
*   **Translation:** Shifts geometric locations along $(x, y)$ planes using an affine transform matrix. Ideal for data augmentation or resetting document baselines.
*   **Rotation:** Shifts orientation matrices around an arbitrary pivot coordinate center using `cv2.getRotationMatrix2D`. Helps correct slight rotation errors.
*   **Scaling:** Resizes base resolution dimensions using standard downsampling (`INTER_AREA`) and upsampling (`INTER_CUBIC`) pixel interpolations.
*   **Affine Transformation:** Alters spatial geometry with 3 independent points mapping non-parallel shear transformations.
*   **Perspective Transformation:** Uses a 4-point mapping structure via homography formulas to resolve complex skew errors, projecting flat 2D views from varying camera angles.

### 2. Image Enhancement Techniques
*   **Brightness Optimization:** Uniformly adds/subtracts scale scalars across pixel vectors to lift shadows or curb heavy exposure.
*   **Contrast Adjustment:** Scales native pixel distributions to widen the dynamic range between deep black text inks and bright parchment backdrops.
*   **Gaussian Blur:** Evaluates pixel weight maps using Gaussian bell curves to eliminate high-frequency digital camera sensor noise.
*   **Median Blur:** Swaps target pixel arrays with evaluated block medians, eliminating disruptive salt-and-pepper noise components without losing entire edge definitions.
*   **Bilateral Filtering:** Combines range and spatial gaussian variations to preserve stark text-to-paper margins while removing color noise across single-tone paper backgrounds.
*   **Image Sharpening:** Employs high-pass spatial laplacian filtering to restore fine line detail across letters and characters.

---

## 📈 Quality Vector Evaluation Impact
The **Perspective Transformation** had the single highest impact on functional document readability. Before applying the homography calculation, text alignments were geometrically distorted due to skewed camera angles, making them completely unreadable for standard Optical Character Recognition (OCR) systems. Correcting the perspective layout realigns the text lines symmetrically to the screen axis, optimizing readability.

---

## 🛑 Implementation Challenges Faced
1.  **Contour Inconsistencies:** Low lighting and low contrast between the document edge and the table surface caused Canny edge detection algorithms to miss document borders. This was mitigated by adding robust fallbacks and giving users slider options to adjust parameters manually within Gradio.
2.  **Pixel Data Overflow:** Manual brightness addition risks pushing pixel brightness bounds beyond the standard `uint8` limits ($255$), producing harsh color wrapping errors. This was resolved using `cv2.convertScaleAbs()`, which caps boundaries safely at $0$ and $255$.
