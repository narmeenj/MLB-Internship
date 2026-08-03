import os
import glob
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_images", "edge_detection")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_bgr(gray_like):
    img = cv2.convertScaleAbs(gray_like)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    return img


def detect_edges(image_path, out_dir):
    name = os.path.splitext(os.path.basename(image_path))[0]
    original = cv2.imread(image_path)
    if original is None:
        print(f"  [skip] could not read {image_path}")
        return

    h, w = original.shape[:2]
    if w > 500:
        scale = 500 / w
        original = cv2.resize(original, (500, int(h * scale)))

    # 1. Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # 2. Gaussian Blur before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3a. Sobel operator
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_mag = np.clip(sobel_mag, 0, 255)

    # 3b. Laplacian operator
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))

    # 3c. Canny edge detection (median-based auto thresholds)
    median_val = np.median(blurred)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    canny = cv2.Canny(blurred, lower, upper)

    cv2.imwrite(os.path.join(out_dir, f"{name}_1_gray.jpg"), gray)
    cv2.imwrite(os.path.join(out_dir, f"{name}_2_blurred.jpg"), blurred)
    cv2.imwrite(os.path.join(out_dir, f"{name}_3_sobel.jpg"), to_bgr(sobel_mag))
    cv2.imwrite(os.path.join(out_dir, f"{name}_4_laplacian.jpg"), to_bgr(laplacian))
    cv2.imwrite(os.path.join(out_dir, f"{name}_5_canny_low{lower}_high{upper}.jpg"), canny)

    tiles = [
        label(original, "Original"),
        label(to_bgr(gray), "Grayscale"),
        label(to_bgr(sobel_mag), "Sobel"),
        label(to_bgr(laplacian), "Laplacian"),
        label(to_bgr(canny), f"Canny ({lower},{upper})"),
    ]
    comparison = np.hstack(tiles)
    cv2.imwrite(os.path.join(out_dir, f"{name}_COMPARISON.jpg"), comparison)
    print(f"  [ok] {name}  (Canny thresholds: {lower}-{upper})")


def main():
    images = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )
    if not images:
        print(f"No images found in {INPUT_DIR}. Add some images to that folder first.")
        return
    print(f"Found {len(images)} image(s). Running edge detection...\n")
    for path in images:
        detect_edges(path, OUTPUT_DIR)
    print(f"\nDone. Results saved to: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()