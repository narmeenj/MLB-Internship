import os
import glob
import cv2
import numpy as np
import gradio as gr
from pyngrok import ngrok
from boundary_detection_core import detect_document_boundary

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(SCRIPT_DIR, "input_images")

def bgr_to_rgb(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def run_pipeline(input_image, blur_ksize, morph_ksize, use_auto_canny, canny_low_manual, canny_high_manual):
    if input_image is None:
        return None, None, None, None, None, None, "⬆️ Upload or click a sample image to get started."
    image_bgr = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
    canny_low = None if use_auto_canny else int(canny_low_manual)
    canny_high = None if use_auto_canny else int(canny_high_manual)
    result = detect_document_boundary(
        image_bgr, canny_low=canny_low, canny_high=canny_high,
        blur_ksize=int(blur_ksize), morph_ksize=int(morph_ksize),
    )
    status = (
        "✅ Document boundary detected successfully!"
        if result["found_quad"] else
        "⚠️ Couldn't find a clean 4-sided boundary — showing best guess. Try the sliders in Advanced Settings."
    )
    status += f"\nCanny thresholds used: {result['canny_thresholds'][0]} – {result['canny_thresholds'][1]}"
    warped_rgb = bgr_to_rgb(result["warped"]) if result["warped"] is not None else None
    return (
        bgr_to_rgb(result["gray"]),
        bgr_to_rgb(result["blurred"]),
        bgr_to_rgb(result["edges"]),
        bgr_to_rgb(result["morphology"]),
        bgr_to_rgb(result["boundary"]),
        warped_rgb,
        status,
    )

def get_sample_paths():
    paths = sorted(
        glob.glob(os.path.join(SAMPLES_DIR, "*.jpg")) +
        glob.glob(os.path.join(SAMPLES_DIR, "*.jpeg")) +
        glob.glob(os.path.join(SAMPLES_DIR, "*.png"))
    )
    return paths

CUSTOM_CSS = """
#header {text-align: center; padding: 10px 0 4px 0;}
#header h1 {margin-bottom: 4px;}
.gr-button-primary {font-weight: 600 !important;}
#status-box textarea {font-weight: 600; font-size: 15px;}
"""

with gr.Blocks(title="Document Boundary Detection Tool", theme=gr.themes.Soft(primary_hue="indigo"), css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="header"):
        gr.Markdown(
            """
            # 📄 Document Boundary Detection Tool
            Upload a photo of a document, and this tool will automatically find and outline its edges — even if it's tilted, shadowed, or a little blurry.
            """
        )
    sample_paths = get_sample_paths()
    if sample_paths:
        gr.Markdown("### 🖼️ Try one of these sample documents")
        gallery = gr.Gallery(
            value=sample_paths, columns=6, height=160, show_label=False,
            allow_preview=False, object_fit="cover",
        )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="1️⃣ Input Document Image", type="numpy", height=300)
            with gr.Accordion("⚙️ Advanced settings", open=False):
                blur_ksize = gr.Slider(1, 15, value=5, step=2, label="Gaussian Blur kernel size")
                morph_ksize = gr.Slider(1, 15, value=5, step=2, label="Morphology kernel size")
                use_auto_canny = gr.Checkbox(value=True, label="Auto Canny thresholds")
                canny_low_manual = gr.Slider(0, 255, value=75, step=1, label="Manual Canny lower")
                canny_high_manual = gr.Slider(0, 255, value=150, step=1, label="Manual Canny upper")
            run_btn = gr.Button("🔍 Detect Boundary", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False, elem_id="status-box")
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("**Pipeline stages**")
                with gr.Row():
                    gray_out = gr.Image(label="2️⃣ Grayscale", height=200)
                    blur_out = gr.Image(label="3️⃣ Gaussian Blur", height=200)
                with gr.Row():
                    edges_out = gr.Image(label="4️⃣ Canny Edges", height=200)
                    morph_out = gr.Image(label="5️⃣ Morphological Cleanup", height=200)
            with gr.Group():
                gr.Markdown("**Final result**")
                with gr.Row():
                    boundary_out = gr.Image(label="6️⃣ Detected Boundary", height=260)
                    warped_out = gr.Image(label="7️⃣ Scanned / Straightened (bonus)", height=260)
    def load_gallery_image(evt: gr.SelectData):
        return sample_paths[evt.index]
    if sample_paths:
        gallery.select(fn=load_gallery_image, inputs=None, outputs=input_image)
    inputs = [input_image, blur_ksize, morph_ksize, use_auto_canny, canny_low_manual, canny_high_manual]
    outputs = [gray_out, blur_out, edges_out, morph_out, boundary_out, warped_out, status_box]
    run_btn.click(fn=run_pipeline, inputs=inputs, outputs=outputs)
    input_image.change(fn=run_pipeline, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    public_url = ngrok.connect(7860)
    print(f"Public URL: {public_url}")
    demo.launch(server_port=7860)