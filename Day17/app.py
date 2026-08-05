import os
import sys
import cv2
import gradio as gr
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from shape_detection import ShapeDetector

detector = ShapeDetector(min_area=200)


def gradio_pipeline(input_image):
    if input_image is None:
        return None, None, pd.DataFrame()

    bgr_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)

    contour_out, final_out, metrics = detector.process_image(bgr_img)

    contour_rgb = cv2.cvtColor(contour_out, cv2.COLOR_BGR2RGB)
    final_rgb = cv2.cvtColor(final_out, cv2.COLOR_BGR2RGB)

    df_metrics = pd.DataFrame(metrics)

    return contour_rgb, final_rgb, df_metrics


input_images_dir = os.path.join(current_dir, "input_images")
example_list = []

if os.path.exists(input_images_dir):
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    all_files = sorted([
        f for f in os.listdir(input_images_dir) if f.lower().endswith(valid_extensions)
    ])
    
    filtered_files = [f for f in all_files if "image_10" not in f and "shape_10" not in f]
    
    for file in filtered_files[:3]:
        example_list.append([os.path.join(input_images_dir, file)])

with gr.Blocks(title="AI Shape Detection Engine") as demo:
    gr.Markdown("# Geometric Shape Detection System")
    gr.Markdown(
        "Select a quick sample from the catalog or upload a custom visual pattern canvas below."
    )

    with gr.Row():
        with gr.Column():
            input_view = gr.Image(type="numpy", label="Upload Source Image")
            process_btn = gr.Button("Analyze Image", variant="primary")
        with gr.Column():
            contour_view = gr.Image(type="numpy", label="Step 1: Contour Boundary Map")

    with gr.Row():
        with gr.Column():
            final_view = gr.Image(
                type="numpy", label="Step 2: Classified Shapes with Labels"
            )
        with gr.Column():
            gr.Markdown("### Extracted Geometric Analysis Logs")
            metrics_table = gr.Dataframe(
                headers=[
                    "Shape #",
                    "Type",
                    "Area (px²)",
                    "Perimeter (px)",
                    "Vertices",
                ]
            )

    if example_list:
        gr.Markdown("### Try out these sample images:")
        gr.Examples(
            examples=example_list,
            inputs=[input_view],
            outputs=[contour_view, final_view, metrics_table],
            fn=gradio_pipeline,
            cache_examples=False,
        )

    process_btn.click(
        fn=gradio_pipeline,
        inputs=[input_view],
        outputs=[contour_view, final_view, metrics_table],
    )

if __name__ == "__main__":
    demo.launch()
