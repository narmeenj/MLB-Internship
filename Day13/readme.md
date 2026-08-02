# Day 13: Object Detection Mini Project using YOLOv8

## 🧠 Conceptual Overview
* **What is Object Detection?** It is a computer vision task that involves identifying both the presence of objects within an image and pinning down their exact location.
* **How is it different from Image Classification?** Image classification outputs a single label for the entire image (e.g., "traffic"). Object detection outputs *what* is in the image and *where* it is by drawing bounding boxes around multiple distinct items (e.g., "3 cars, 1 traffic light").
* **What is YOLO?** "You Only Look Once" is a state-of-the-art, real-time object detection algorithm. It processes the entire image in a single forward pass through the neural network, making it incredibly fast and highly accurate.

## 📊 Dataset & Inference Details
* **Dataset Used:** [e.g., Vehicle Detection / Fruit Detection via Kaggle/Roboflow]
* **Objects Detected:** [e.g., car, truck, bus, motorcycle]

## 🔍 Model Predictions & Observations
* **Successes:** The model easily detected [e.g., clear, large vehicles in daylight with high confidence (> 85%)].
* **Limitations/Flaws:** It struggled slightly with [e.g., tiny vehicles in the background or heavily occluded/overlapping objects, dropping confidence scores down to 40%].
* **Insight:** Since we used a pre-trained COCO model without custom fine-tuning, it performs exceptionally well on common daily objects but might miss specific niche classes.

## 📱 How the Gradio App Works
1. The user uploads an image via the web interface (`gr.Image` input node).
2. The image matrix is passed into the `detect_objects` Python function as a NumPy array.
3. The pre-trained `yolov8n.pt` model executes inference on the array.
4. The `.plot()` function from the Ultralytics library overlays the bounding boxes, confidence tags, and class names onto the image matrix.
5. The processed matrix is rendered back on the web UI layout.
