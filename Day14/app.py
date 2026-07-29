import os
import sys
import traceback

import cv2
import numpy as np
import gradio as gr

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "image_toolkit"))
import toolkit  # noqa: E402

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "sample_images")

SAMPLE_IMAGES = {
    "Landscape": os.path.join(SAMPLE_DIR, "landscape.jpg"),
    "Person": os.path.join(SAMPLE_DIR, "person.jpg"),
    "Vehicle": os.path.join(SAMPLE_DIR, "vehicle.jpg"),
    "Document": os.path.join(SAMPLE_DIR, "document.jpg"),
    "Object": os.path.join(SAMPLE_DIR, "object.jpg"),
}

OPERATIONS = [
    "Grayscale",
    "Resize",
    "Rotate",
    "Flip",
    "Crop",
    "Draw Shapes",
    "Add Text",
    "Brightness / Contrast",
    "BGR vs RGB Comparison",
]


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb_for_display(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return None
    if len(img_bgr.shape) == 2:  # grayscale -> still fine for gr.Image
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_sample(name: str):
    path = SAMPLE_IMAGES.get(name)
    if path is None:
        return None
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    return bgr_to_rgb_for_display(img_bgr)


def update_controls(operation: str):
    return {
        resize_w: gr.update(visible=operation == "Resize"),
        resize_h: gr.update(visible=operation == "Resize"),
        rotate_angle: gr.update(visible=operation == "Rotate"),
        flip_mode: gr.update(visible=operation == "Flip"),
        crop_x: gr.update(visible=operation == "Crop"),
        crop_y: gr.update(visible=operation == "Crop"),
        crop_w: gr.update(visible=operation == "Crop"),
        crop_h: gr.update(visible=operation == "Crop"),
        text_input: gr.update(visible=operation == "Add Text"),
        text_x: gr.update(visible=operation == "Add Text"),
        text_y: gr.update(visible=operation == "Add Text"),
        brightness_slider: gr.update(visible=operation == "Brightness / Contrast"),
        contrast_slider: gr.update(visible=operation == "Brightness / Contrast"),
    }


def process_image(image_rgb, operation, resize_w_v, resize_h_v, rotate_angle_v,
                   flip_mode_v, crop_x_v, crop_y_v, crop_w_v, crop_h_v,
                   text_v, text_x_v, text_y_v, brightness_v, contrast_v):
    try:
        if image_rgb is None:
            return None, None, "Please upload an image before proceeding."

        if not isinstance(image_rgb, np.ndarray) or image_rgb.ndim not in (2, 3):
            return None, None, "Invalid file. Please upload a valid image."

        img_bgr = rgb_to_bgr(image_rgb) if image_rgb.ndim == 3 else image_rgb

        if operation is None:
            return None, None, "Please choose an operation."

        if operation == "Grayscale":
            processed = toolkit.convert_grayscale(img_bgr)

        elif operation == "Resize":
            processed = toolkit.resize_img(img_bgr, int(resize_w_v), int(resize_h_v))

        elif operation == "Rotate":
            processed = toolkit.rotate_any(img_bgr, float(rotate_angle_v))

        elif operation == "Flip":
            processed = toolkit.flip_img(img_bgr, flip_mode_v)

        elif operation == "Crop":
            processed = toolkit.crop_image(img_bgr, int(crop_x_v), int(crop_y_v),
                                            int(crop_w_v), int(crop_h_v))

        elif operation == "Draw Shapes":
            processed = toolkit.draw_shapes(img_bgr)

        elif operation == "Add Text":
            if not text_v:
                return None, None, "Please enter some text to add."
            processed = toolkit.add_text(img_bgr, text_v, (int(text_x_v), int(text_y_v)))

        elif operation == "Brightness / Contrast":
            processed = toolkit.adjust_brightness_contrast(
                img_bgr, int(brightness_v), int(contrast_v))

        elif operation == "BGR vs RGB Comparison":
            rgb_version = toolkit.bgr_to_rgb(img_bgr)
            processed = np.hstack([img_bgr, rgb_version])

        else:
            return None, None, "Unknown operation selected."

        preview = toolkit.side_by_side(img_bgr, processed)
        preview_rgb = bgr_to_rgb_for_display(preview)
        processed_rgb = bgr_to_rgb_for_display(processed)

        return preview_rgb, processed_rgb, "Image processed successfully!"

    except ValueError as e:
        return None, None, f"{e}"
    except Exception:
        traceback.print_exc()
        return None, None, ("Something went wrong while processing the "
                             "image. Please check your inputs and try again.")


def save_processed(processed_rgb):
    try:
        if processed_rgb is None:
            return None, "Nothing to save yet - process an image first."
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "processed_output.png")
        bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR) \
            if processed_rgb.ndim == 3 else processed_rgb
        ok = cv2.imwrite(out_path, bgr)
        if not ok:
            return None, "Failed to save the image."
        return out_path, "Saved! Use the download link below."
    except Exception:
        traceback.print_exc()
        return None, "Could not save the image due to an unexpected error."


# UI

TITLE = "Image Processing Toolkit"
DESCRIPTION = """
Upload your own image **or** pick one of the sample images below, choose an
operation, and click **Process**. You'll see the original and processed
image side by side, and can download the result.

**How to use:**
1. Upload an image, or click a sample image to load it.
2. Pick an operation from the dropdown (extra options will appear if needed).
3. Click **Process Image**.
4. Click **Prepare Download** to get a downloadable file of the result.

Supported operations: Grayscale, Resize, Rotate, Flip, Crop, Draw Shapes,
Add Text, Brightness/Contrast adjustment, and a BGR vs RGB comparison.
"""

with gr.Blocks(title="Image Processing Toolkit") as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Input Image", type="numpy")

            gr.Markdown("**Or try a sample image:**")
            with gr.Row():
                sample_buttons = []
                for name in SAMPLE_IMAGES:
                    btn = gr.Button(name, size="sm")
                    sample_buttons.append((btn, name))

            operation = gr.Dropdown(choices=OPERATIONS, label="Operation",
                                     value="Grayscale")
            resize_w = gr.Number(label="Resize width (px)", value=300, visible=False)
            resize_h = gr.Number(label="Resize height (px)", value=200, visible=False)
            rotate_angle = gr.Slider(-180, 180, value=90, label="Rotation angle (deg)",
                                      visible=False)
            flip_mode = gr.Radio(["horizontal", "vertical", "both"],
                                  value="horizontal", label="Flip mode", visible=False)
            crop_x = gr.Number(label="Crop X (top-left)", value=0, visible=False)
            crop_y = gr.Number(label="Crop Y (top-left)", value=0, visible=False)
            crop_w = gr.Number(label="Crop width", value=100, visible=False)
            crop_h = gr.Number(label="Crop height", value=100, visible=False)
            text_input = gr.Textbox(label="Text to add", visible=False)
            text_x = gr.Number(label="Text X position", value=50, visible=False)
            text_y = gr.Number(label="Text Y position", value=50, visible=False)
            brightness_slider = gr.Slider(-100, 100, value=0, label="Brightness",
                                           visible=False)
            contrast_slider = gr.Slider(-100, 100, value=0, label="Contrast",
                                         visible=False)

            process_btn = gr.Button("Process Image", variant="primary")

        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False)
            preview_output = gr.Image(label="Original vs Processed (side by side)")
            processed_output = gr.Image(label="Processed Image", visible=True)
            save_btn = gr.Button("Prepare Download")
            download_file = gr.File(label="Download processed image")

    for btn, name in sample_buttons:
        btn.click(fn=lambda n=name: load_sample(n), inputs=None, outputs=image_input)

    operation.change(
        fn=update_controls,
        inputs=operation,
        outputs=[resize_w, resize_h, rotate_angle, flip_mode, crop_x, crop_y,
                 crop_w, crop_h, text_input, text_x, text_y,
                 brightness_slider, contrast_slider],
    )

    process_btn.click(
        fn=process_image,
        inputs=[image_input, operation, resize_w, resize_h, rotate_angle,
                flip_mode, crop_x, crop_y, crop_w, crop_h, text_input,
                text_x, text_y, brightness_slider, contrast_slider],
        outputs=[preview_output, processed_output, status],
    )

    save_btn.click(
        fn=save_processed,
        inputs=processed_output,
        outputs=[download_file, status],
    )

if __name__ == "__main__":
    demo.launch()
