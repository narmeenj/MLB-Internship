import os
import glob
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_images", "morphological_operations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def apply_morphology(image_path, out_dir):
    name = os.path.splitext(os.path.basename(image_path))[0]
    original = cv2.imread(image_path)
    if original is None:
        print(f"  [skip] could not read {image_path}")
        return

    h, w = original.shape[:2]
    if w > 500:
        scale = 500 / w
        original = cv2.resize(original, (500, int(h * scale)))

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    median_val = np.median(blurred)
    lower, upper = int(max(0, 0.66 * median_val)), int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, lower, upper)

    erosion = cv2.erode(edges, KERNEL, iterations=1)
    dilation = cv2.dilate(edges, KERNEL, iterations=1)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, KERNEL)
    closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, KERNEL)
    gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, KERNEL)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, KERNEL)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, KERNEL)

    stage_map = {
        "1_edges_input": edges, "2_erosion": erosion, "3_dilation": dilation,
        "4_opening": opening, "5_closing": closing, "6_gradient": gradient,
        "7_tophat": tophat, "8_blackhat": blackhat,
    }
    for suffix, img in stage_map.items():
        cv2.imwrite(os.path.join(out_dir, f"{name}_{suffix}.jpg"), img)

    row1 = np.hstack([
        label(original, "Original"),
        label(to_bgr(edges), "Before (Canny edges)"),
        label(to_bgr(erosion), "Erosion"),
        label(to_bgr(dilation), "Dilation"),
    ])
    row2 = np.hstack([
        label(to_bgr(opening), "Opening"),
        label(to_bgr(closing), "Closing"),
        label(to_bgr(gradient), "Gradient"),
        label(to_bgr(cv2.addWeighted(tophat, 1.0, blackhat, 1.0, 0)), "TopHat+BlackHat"),
    ])
    comparison = np.vstack([row1, row2])
    cv2.imwrite(os.path.join(out_dir, f"{name}_COMPARISON.jpg"), comparison)
    print(f"  [ok] {name}")


def main():
    images = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )
    if not images:
        print(f"No images found in {INPUT_DIR}.")
        return
    print(f"Found {len(images)} image(s). Running morphological operations...\n")
    for path in images:
        apply_morphology(path, OUTPUT_DIR)
    print(f"\nDone. Results saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()