# 📄 Lab Write-Up: Document Boundary Detection Tool

## 🔍 1. Edge Detection Comparison
* **Sobel Operator**: Calculates first-order image derivatives to locate horizontal and vertical intensity changes separately. It highlights directional edges well but creates thick lines and misses multi-directional corner joints cleanly.
* **Laplacian Operator**: Computes second-order spatial derivatives to capture rapid grayscale changes across all directions at once. While it maps full perimeters, it amplifies random background noise, resulting in a speckled output.
* **Canny Edge Detection**: A multi-stage architecture that suppresses structural noise using a Gaussian filter, sharpens boundaries with non-maximum suppression, and bridges broken segments via hysteresis thresholding. This makes it the gold standard for clean edge profiles.

---

## 🛠️ 2. Purpose of Morphological Operations
* **Erosion**: Shrinks foreground objects by trimming outer pixel boundaries. It is used to eliminate tiny, isolated background noise dots.
* **Dilation**: Expands foreground object boundaries by adding pixels to outer borders. It is used to bridge thin line breaks and close minor structural cracks.
* **Opening**: Performs an Erosion step followed immediately by a Dilation step. It removes small foreground clutter (like text characters) while keeping the original object scale intact.
* **Closing**: Performs a Dilation step followed immediately by an Erosion step. It fills background holes and fuses broken segments along object lines without altering their thickness.
* **Morphological Gradient**: Computes the absolute mathematical difference between a dilation and erosion pass. It creates a stark structural outline map of targeted objects.
* **Top Hat**: Subtracts an image's Opening result from the original image. It isolates bright localized details smaller than the structural kernel element.
* **Black Hat**: Subtracts the original image from its Closing result. It isolates dark localized patches (like deep internal voids or shadows) smaller than the structural kernel matrix.

---

## 🚀 3. Best Performing Combination of Techniques
The most stable computer vision pipeline across diverse document conditions consisted of the following sequence:
1. **Grayscale Conversion**: Eliminates color variations to focus purely on structural density gradients.
2. **Gaussian Blur (5x5)**: Blends out text block layouts and page artifacts to suppress interior edge noise.
3. **Canny Edge Detection**: Leverages auto-calculated median thresholds to extract highly accurate, crisp document border paths.
4. **Morphological Closing**: Uses a rectangular 5x5 structural element kernel to seal line segments broken by poor lighting or glare.
5. **Largest External Contour Filtering**: Isolates the single largest continuous perimeter loop to extract the primary document bounding box layout.

---

## ⚠️ 4. Production Engineering Challenges Faced
* **Low Background Contrast**: Light-colored white printer sheets resting on light wooden tables produced weak gradient loops, making standard edge thresholds fail.
* **Ambient Shadows & Uneven Lighting**: Overhead phone photography cast dark gradient shifts over document corners, breaking continuous edge lines and confusing standard contour filters.
* **Internal Text Bleed Noise**: Bold headers and dense paragraph clusters registered strong localized contrast edges, which created false internal boundary boxes inside the main paper borders.
