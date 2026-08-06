import gradio as gr
import cv2
import os
import numpy as np

def process_video_interface(video_path, blur_kernel, low_threshold, high_threshold):
    if not video_path:
        return None, "Please upload a video file first."
        
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = "output_processed_hf.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height), True)
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        k_size = int(blur_kernel)
        if k_size % 2 == 0:
            k_size += 1
            
        blurred = cv2.GaussianBlur(gray, (k_size, k_size), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Convert grayscale/edge channels back to 3-channel BGR for a clean side-by-side horizontal stack
        gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edges_3c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        combined_frame = np.hstack((gray_3c, edges_3c))
        writer.write(combined_frame)
        
    capture.release()
    writer.release()
    
    metadata_log = f"Processing Complete!\nResolution: {width}x{height}\nFPS: {fps}\nApplied Parameters:\n- Gaussian Blur Kernel: {k_size}x{k_size}\n- Canny Lower Bound: {low_threshold}\n- Canny Upper Bound: {high_threshold}"
    return output_path, metadata_log

with gr.Blocks(title="Real-Time Video Processing Tool") as demo:
    gr.Markdown("#Day 18: Video Processing & Structure Detection Tool")
    gr.Markdown("Upload any short `.mp4` video file to process it through our Computer Vision pipeline. You can use the interactive sliders below to dynamically tune the thresholds!")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Step 1: Upload Input Video")
            
            gr.Markdown("###Pipeline Parameters Adjustment")
            blur_slider = gr.Slider(minimum=1, maximum=15, step=2, value=5, label="Gaussian Blur Kernel Size")
            low_thresh = gr.Slider(minimum=0, maximum=255, step=1, value=50, label="Canny Lower Threshold Bound")
            high_thresh = gr.Slider(minimum=0, maximum=255, step=1, value=150, label="Canny Upper Threshold Bound")
            
            submit_btn = gr.Button("Execute Processing Pipeline", variant="primary")
            
        with gr.Column():
            video_output = gr.Video(label="Processed Output (Grayscale vs Canny Edges)")
            diagnostics_log = gr.Textbox(label="System Media Metadata Diagnostics", lines=6)
            
    submit_btn.click(
        fn=process_video_interface,
        inputs=[video_input, blur_slider, low_thresh, high_thresh],
        outputs=[video_output, diagnostics_log]
    )
    
    gr.Markdown("###Pipeline Reference Samples")
    gr.Markdown("Below are static image matrix examples showing how a single frame transforms through this exact pipeline layout:")
    
    with gr.Row():
        gr.Image(value="https://githubusercontent.com", label="Step 1: Original Sample Matrix Frame")
        gr.Image(value="https://githubusercontent.com", label="Step 2: Extracted Canny Edges Profile Output")

demo.queue().launch()
