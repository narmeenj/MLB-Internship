import os
import sys
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from shape_detection import ShapeDetector


def run_challenge():
    input_folder = os.path.join(current_dir, "input_images")
    output_folder = os.path.join(current_dir, "output_images/output_challenge")

    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        os.makedirs(input_folder, exist_ok=True)
        print(f"Created folder: {input_folder}. Please add images.")
        return

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No images found in {input_folder}")
        return

    detector = ShapeDetector(min_area=200)
    print(f"Processing {len(image_files)} images...")

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read {filename}. Skipping.")
            continue

        print(f"Processing: {filename}")
        base_name = os.path.splitext(filename)[0]

        try:
            contour_img, final_img, stats = detector.process_image(img)

            cv2.imwrite(os.path.join(output_folder, f"{base_name}_contour.png"), contour_img)
            cv2.imwrite(os.path.join(output_folder, f"{base_name}_final.png"), final_img)
            print(f"Saved results for {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

    print("Batch processing complete.")
    print()


if __name__ == "__main__":
    run_challenge()
