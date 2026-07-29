from __future__ import annotations
import os
import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at '{image_path}'. "
                          f"Make sure the file exists and is a valid image.")
    return img


def read_image(image_path: str, verbose: bool = True) -> np.ndarray:
    img = load_image(image_path)

    if verbose:
        shape_img = img.shape
        height_img = shape_img[0]
        width_img = shape_img[1]
        channels = shape_img[2] if len(shape_img) > 2 else 1
        file_size_kb = round(os.path.getsize(image_path) / 1024, 2)

        print("______________________________________________")
        print("____Analysis____", os.path.basename(image_path))
        print("______________________________________________")
        print("Height:", height_img, "pixels")
        print("Width:", width_img, "pixels")
        print("Color Channels:", channels, "BGR")
        print("File Size:", file_size_kb, "KB")
        print("______________________________________________")
        print()

    return img


def save_image(img: np.ndarray, path: str) -> None:
    ok = cv2.imwrite(path, img)
    if not ok:
        raise ValueError(f"Failed to save image to '{path}'.")


def convert_grayscale(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        raise ValueError("Input image is empty.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


to_grayscale = convert_grayscale



def resize_img(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive integers.")
    return cv2.resize(img, (int(width), int(height)),
                       interpolation=cv2.INTER_LINEAR)


resize_image = resize_img


def resize_multiple(img: np.ndarray) -> dict:
    sizes = {
        "small_200x200": (200, 200),
        "medium_400x400": (400, 400),
        "large_800x800": (800, 800),
        "wide_640x360": (640, 360),
    }
    return {name: cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            for name, (w, h) in sizes.items()}


def resize_by_scale(img: np.ndarray, scale_percent: float) -> np.ndarray:
    if scale_percent <= 0:
        raise ValueError("Scale percent must be positive.")
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale_percent / 100))
    new_h = max(1, int(h * scale_percent / 100))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def crop_region(img: np.ndarray, start_x: int, start_y: int,
                 end_x: int, end_y: int) -> np.ndarray:
    h, w = img.shape[:2]
    if start_x < 0 or start_y < 0 or end_x <= start_x or end_y <= start_y:
        raise ValueError("Invalid crop coordinates.")
    if end_x > w or end_y > h:
        raise ValueError("Crop region goes outside the image bounds.")
    return img[start_y:end_y, start_x:end_x].copy()


def crop_image(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    return crop_region(img, x, y, x + w, y + h)


def crop_regions(img: np.ndarray) -> dict:
    h, w = img.shape[:2]
    return {
        "upper": img[0:h // 2, 0:w],
        "lower": img[h // 2:h, 0:w],
        "left": img[0:h, 0:w // 2],
        "right": img[0:h, w // 2:w],
        "center": img[h // 4:3 * h // 4, w // 4:3 * w // 4],
        "top_left": img[0:h // 2, 0:w // 2],
        "top_right": img[0:h // 2, w // 2:w],
        "bottom_left": img[h // 2:h, 0:w // 2],
        "bottom_right": img[h // 2:h, w // 2:w],
    }


def rotate_img(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Only 90, 180, 270 degrees are supported by "
                          "rotate_img() - use rotate_any() for a free angle.")


def rotate_any(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(img, matrix, (new_w, new_h))

def flip_img(img: np.ndarray, direction: str) -> np.ndarray:
    """Flip the image. direction: 'horizontal', 'vertical', or 'both'."""
    direction = direction.lower()
    if direction == "horizontal":
        return cv2.flip(img, 1)
    elif direction == "vertical":
        return cv2.flip(img, 0)
    elif direction == "both":
        return cv2.flip(img, -1)
    else:
        raise ValueError("Only 'horizontal', 'vertical' and 'both' flips "
                          "are supported.")


flip_image = flip_img


def draw_shapes(img: np.ndarray) -> np.ndarray:
    canvas = img.copy()
    h, w = canvas.shape[:2]

    cv2.rectangle(canvas, (20, 20), (150, 150), (0, 255, 0), 3)
    cv2.circle(canvas, (w // 2, h // 2), 60, (255, 0, 0), -1)
    cv2.line(canvas, (30, h - 50), (w - 30, h - 50), (0, 0, 255), 4)

    polygon = np.array([[10, h - 10], [50, h - 40], [90, h - 10]], np.int32)
    cv2.polylines(canvas, [polygon.reshape((-1, 1, 2))],
                  isClosed=True, color=(0, 255, 255), thickness=3)

    return canvas


def add_text(img: np.ndarray, text: str, position=None,
             font_scale: float = None, color=(255, 255, 255),
             thickness: int = 2) -> np.ndarray:
    if not text:
        raise ValueError("Text cannot be empty.")
    canvas = img.copy()
    h, w = canvas.shape[:2]

    if font_scale is None:
        font_scale = max(0.5, w / 900.0)
    if position is None:
        position = (20, h - 15)

    cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return canvas

def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0,
                                contrast: int = 0) -> np.ndarray:

    brightness = int(np.clip(brightness, -100, 100))
    contrast = int(np.clip(contrast, -100, 100))
    alpha = 1 + (contrast / 100.0)
    beta = brightness
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    if img1 is None or img2 is None:
        raise ValueError("Both images must be provided for comparison.")

    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    h1 = img1.shape[0]
    h2, w2 = img2.shape[:2]
    if h2 != h1:
        scale = h1 / h2
        img2 = cv2.resize(img2, (int(w2 * scale), h1))

    return np.hstack([img1, img2])
