import cv2
import numpy as np
import os
import sys


def adjust_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + value, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_contrast(img, alpha=1.3):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def gaussian_blur(img, ksize=5, sigma=0):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def median_blur(img, ksize=5):
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def sharpen_image(img, amount=1.0):
    kernel = np.array([[0, -1, 0],
                        [-1, 5 + amount, -1],
                        [0, -1, 0]]) / (1 + amount)
    return cv2.filter2D(img, -1, kernel)


def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise_document(img):
    return bilateral_filter(img, d=9, sigma_color=75, sigma_space=75)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "output_images", "enhancements")
    os.makedirs(out_dir, exist_ok=True)

    input_dir = os.path.join(os.path.dirname(__file__), "input_images")
    own_images = []
    if os.path.isdir(input_dir):
        own_images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    src_candidates = [
        sys.argv[1] if len(sys.argv) > 1 else None,
        os.path.join(input_dir, own_images[0]) if own_images else None,
        os.path.join(os.path.dirname(__file__), "output_images", "transformations", "demo_source.jpg"),
    ]
    src = next((p for p in src_candidates if p and os.path.exists(p)), None)

    if src is None:
        raise SystemExit("No source image found. Run image_transformation.py first, or add an image to input_images/.")

    image = cv2.imread(src)
    print(f"Using source image: {src}")

    cv2.imwrite(os.path.join(out_dir, "01_original.jpg"), image)
    cv2.imwrite(os.path.join(out_dir, "02_brighter.jpg"), adjust_brightness(image, 50))
    cv2.imwrite(os.path.join(out_dir, "03_darker.jpg"), adjust_brightness(image, -50))
    cv2.imwrite(os.path.join(out_dir, "04_more_contrast.jpg"), adjust_contrast(image, 1.5))
    cv2.imwrite(os.path.join(out_dir, "05_less_contrast.jpg"), adjust_contrast(image, 0.7))
    cv2.imwrite(os.path.join(out_dir, "06_gaussian_blur.jpg"), gaussian_blur(image, 7))
    cv2.imwrite(os.path.join(out_dir, "07_median_blur.jpg"), median_blur(image, 7))
    cv2.imwrite(os.path.join(out_dir, "08_bilateral_filter.jpg"), bilateral_filter(image))
    cv2.imwrite(os.path.join(out_dir, "09_sharpened.jpg"), sharpen_image(image, 1.0))
    cv2.imwrite(os.path.join(out_dir, "10_grayscale.jpg"), to_grayscale(image))

    print(f"All enhancement examples saved to: {out_dir}")