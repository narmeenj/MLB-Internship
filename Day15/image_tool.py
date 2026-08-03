import cv2
import numpy as np
import gradio as gr

from image_transformation import auto_detect_document_corners, perspective_transform
from image_enhancement import adjust_brightness, adjust_contrast, bilateral_filter, sharpen_image, to_grayscale


def enhance_document(image, brightness, contrast, sharpen_amount, apply_grayscale, correct_perspective):
    if image is None:
        return None, None

    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    corrected = img_bgr
    if correct_perspective:
        corners = auto_detect_document_corners(img_bgr)
        if corners is not None:
            corrected = perspective_transform(img_bgr, corners)

    perspective_preview = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)

    denoised = bilateral_filter(corrected)
    adjusted = adjust_brightness(denoised, brightness)
    adjusted = adjust_contrast(adjusted, contrast)
    sharpened = sharpen_image(adjusted, sharpen_amount)

    if apply_grayscale:
        final = to_grayscale(sharpened)
        final_rgb = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)
    else:
        final_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)

    return perspective_preview, final_rgb


with gr.Blocks(title="Document Image Enhancement Tool") as demo:
    gr.Markdown("# 📄 Document Image Enhancement Tool")
    gr.Markdown(
        "Upload a photo of a document (tilted or straight). "
        "The app will straighten it, reduce noise, fix brightness/contrast, and sharpen it."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Document Image", type="numpy")
            correct_perspective = gr.Checkbox(label="Auto-correct perspective", value=True)
            apply_grayscale = gr.Checkbox(label="Convert to grayscale", value=True)
            brightness = gr.Slider(-100, 100, value=20, step=1, label="Brightness")
            contrast = gr.Slider(0.5, 3.0, value=1.3, step=0.1, label="Contrast")
            sharpen_amount = gr.Slider(0.0, 3.0, value=1.0, step=0.1, label="Sharpen Amount")
            run_btn = gr.Button("Enhance Document", variant="primary")

        with gr.Column():
            perspective_output = gr.Image(label="Perspective-Corrected (Step Preview)")
            final_output = gr.Image(label="Final Enhanced Image")

    run_btn.click(
        fn=enhance_document,
        inputs=[input_image, brightness, contrast, sharpen_amount, apply_grayscale, correct_perspective],
        outputs=[perspective_output, final_output],
    )

    gr.Markdown(
        "### Pipeline\n"
        "`Load -> Perspective Correction -> Noise Reduction (Bilateral Filter) -> "
        "Brightness/Contrast -> Sharpening -> (optional) Grayscale`"
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
