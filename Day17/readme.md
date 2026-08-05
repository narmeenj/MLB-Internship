# Day 17 Mini Project: Shape Detection System

## Project Overview
This application uses OpenCV and Gradio to detect, isolate, and classify basic geometric shapes from flat image assets. It tracks boundary perimeters, isolates shape areas, and counts localized vertices using a polygonal reduction algorithm.

## Technical Insights

### 1. What are Contours?
Contours are continuous structural vector curves linking spatial data along a shared peripheral border that has the same pixel color or intensity profile.

### 2. How Contour Detection Works
* **Pre-processing**: Converting frames to grayscale isolates intensity boundaries, and Gaussian Blurring filters out high-frequency noise.
* **Binarization**: Otsu's thresholding dynamically calculates an optimal split value, mapping shapes as pure foreground white pixels against a uniform black backdrop.
* **Border Extraction**: The system uses `cv2.findContours` to scan binary structural transitions, building coordinate matrices that map shape topologies.

### 3. Detectable Geometries
* **Triangles** (3 reduced points)
* **Squares & Rectangles** (4 reduced points, differentiated by bounding aspect ratio variances)
* **Pentagons & Hexagons** (5 & 6 points)
* **Circles & Ovals** (High vertex counts evaluated using a mathematical circularity index: $4 \pi \times \text{Area} / \text{Perimeter}^2$)

## Challenges Faced & Fixes Applied
* **Shadow Artifacts & Real Photo Noise**: Standard thresholding caused rough shape edges, which led to incorrect vertex counts. 
  * *Fix*: Replaced simple global thresholding with **Otsu’s Thresholding** paired with a preceding **Gaussian Blur** pass to clean up edge variations.
* **Label Placement Errors**: Centroid coordinate positions calculated via raw spatial moments sometimes drifted outside shapes with sharp angles.
  * *Fix*: Added a backup bounding-box fallback mechanism (`cv2.boundingRect`) to keep text annotations centered if centroid calculations returned invalid offsets.
