import os
import cv2
import numpy as np

from image_transformation import auto_detect_document_corners, perspective_transform
from image_enhancement import adjust_brightness, adjust_contrast, bilateral_filter, sharpen_image, to_grayscale

INPUT_DIR = os.path.join(os.path.dirname(__file__), "tilted_docs")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_images")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

BRIGHTNESS = 20
CONTRAST = 1.3
SHARPEN_AMOUNT = 1.0
GRAYSCALE = True


def process(img_bgr):
    corners = auto_detect_document_corners(img_bgr)
    corrected = perspective_transform(img_bgr, corners) if corners is not None else img_bgr.copy()

    denoised = bilateral_filter(corrected)
    adjusted = adjust_brightness(denoised, BRIGHTNESS)
    adjusted = adjust_contrast(adjusted, CONTRAST)
    sharpened = sharpen_image(adjusted, SHARPEN_AMOUNT)

    if GRAYSCALE:
        final = cv2.cvtColor(to_grayscale(sharpened), cv2.COLOR_GRAY2BGR)
    else:
        final = sharpened

    return corrected, final


def make_comparison(original, corrected, final, target_h=400):
    def resize_to_h(im, h):
        r = h / im.shape[0]
        return cv2.resize(im, (int(im.shape[1] * r), h))

    imgs = [resize_to_h(original, target_h), resize_to_h(corrected, target_h), resize_to_h(final, target_h)]
    labels = ["Original", "Perspective Corrected", "Final Enhanced"]
    labeled = []
    for im, label in zip(imgs, labels):
        im = im.copy()
        cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(im, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        labeled.append(im)
    return np.hstack(labeled)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMPARISON_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR) or not any(
        f.lower().endswith((".jpg", ".jpeg", ".png")) for f in os.listdir(INPUT_DIR)
    ):
        print(f"No images found in {INPUT_DIR}. Add your tilted document photos there and re-run.")
        return

    count = 0
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Skipping unreadable file: {fname}")
            continue

        corrected, final = process(img)
        name, _ = os.path.splitext(fname)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_corrected.jpg"), corrected)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_enhanced.jpg"), final)

        comparison = make_comparison(img, corrected, final)
        cv2.imwrite(os.path.join(COMPARISON_DIR, f"{name}_comparison.jpg"), comparison)

        count += 1
        print(f"Processed: {fname}")

    print(f"\nDone. Processed {count} image(s).")
    print(f"Outputs: {OUTPUT_DIR}")
    print(f"Comparisons: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()