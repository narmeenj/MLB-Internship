import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os

try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

def detect_objects(input_image):
    if input_image is None:
        return None, "Please upload an image or click a sample image below before proceeding."
    
    if model is None:
        return None, " Model initialization error. Please restart the backend server."
    
    try:
        if not isinstance(input_image, np.ndarray) or input_image.size == 0:
            return None, "Invalid file. Please upload a valid image."
        
        results = model(input_image)
        
        annotated_image = results.plot(bgr=False)
        
        return annotated_image, " Object detection completed successfully!"
        
    except Exception as error:
        return None, f" An unexpected error occurred during processing: {str(error)}"

example_list = [
    ["sample_inputs/car.jpg"],
    ["sample_inputs/fruits.jpg"]
]

verified_examples = [ex for ex in example_list if os.path.exists(ex[0])]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("#  YOLOv8 Real-Time Object Detection Sandbox")
    gr.Markdown(
        """
        ### Application Purpose
        This application utilizes a pre-trained **YOLOv8 (You Only Look Once)** deep learning model to perform real-time object detection. 
        Unlike basic classification, this model identifies *what* objects are present and pinpoints *where* they are by drawing precise bounding boxes around them.
        
        ### How to Use It:
        1. **Upload an Image:** Drag and drop your own photo into the **Input Image** box, OR simply **click on one of the sample images** at the bottom.
        2. **Run Detection:** Click the blue **Detect Objects** button to initialize inference.
        3. **Analyze Results:** View the generated bounding boxes, class labels, and confidence percentage scores alongside the system status notification.
        """
    )
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="numpy", label="Input Image Workspace")
            submit_btn = gr.Button("Detect Objects", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(type="numpy", label="YOLO Detection Result")
            status_output = gr.Textbox(label="System Notification Status", interactive=False)
            
    submit_btn.click(
        fn=detect_objects,
        inputs=image_input,
        outputs=[image_output, status_output]
    )
   
    if verified_examples:
        gr.Markdown("### 📸 Quick Test Samples")
        gr.Examples(
            examples=verified_examples,
            inputs=image_input,
            outputs=[image_output, status_output],
            fn=detect_objects,
            cache_examples=False # Keeps memory low on your local machine
        )
    else:
        gr.Markdown(" *Note: Place sample images in a 'sample_inputs/' directory to display clickable web examples.*")

if __name__ == "__main__":
    demo.launch(server_port=7860)
