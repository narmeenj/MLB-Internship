import os
import cv2
import numpy as np
from datetime import date

# Configuration Paths
input_file = r"E:\Subjects\ML AI Internship\MLB-Internship\Day14\three_kittens.jpg"
output_dir = r"E:\Subjects\ML AI Internship\MLB-Internship\Day14\output_images"
os.makedirs(output_dir, exist_ok=True)

my_name = "Narmeen Javed"
date_tdy = date.today().strftime("%d-%m-%Y")


def read_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image!")
        return None

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


def convert_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_img(img: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)


def crop_region(img: np.ndarray, start_x: int, start_y: int, end_x: int, end_y: int) -> np.ndarray:
    return img[start_y:end_y, start_x:end_x]


def resize_multiple(img: np.ndarray) -> dict:
    """Resizes the image to several different resolutions and returns them all in a dict."""
    sizes = {
        "small_200x200": (200, 200),
        "medium_400x400": (400, 400),
        "large_800x800": (800, 800),
        "wide_640x360": (640, 360),
    }

    resized_images = {}
    for name, (w, h) in sizes.items():
        resized_images[name] = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    return resized_images


def crop_regions(img: np.ndarray) -> dict:
    """Crops the image from several different areas: upper, lower, left, right, and center."""
    h, w = img.shape[:2]

    cropped_images = {
        "upper": img[0 : h // 2, 0:w],
        "lower": img[h // 2 : h, 0:w],
        "left": img[0:h, 0 : w // 2],
        "right": img[0:h, w // 2 : w],
        "center": img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4],
        "top_left": img[0 : h // 2, 0 : w // 2],
        "top_right": img[0 : h // 2, w // 2 : w],
        "bottom_left": img[h // 2 : h, 0 : w // 2],
        "bottom_right": img[h // 2 : h, w // 2 : w],
    }

    return cropped_images


def rotate_img(img: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print("Only 90, 180, 270 degrees supported")
        return img


def flip_img(img: np.ndarray, direction: str) -> np.ndarray:
    if direction.lower() == "horizontal":
        return cv2.flip(img, 1)
    elif direction.lower() == "vertical":
        return cv2.flip(img, 0)
    else:
        print("Only Horizontal and Vertical Flips supported")
        return img


def draw_shapes(img: np.ndarray) -> np.ndarray:
    canvas = img.copy()
    h, w = canvas.shape[:2]

    cv2.rectangle(canvas, (20, 20), (150, 150), (0, 255, 0), 3)
    cv2.circle(canvas, (w // 2, h // 2), 60, (255, 0, 0), -1)
    cv2.line(canvas, (30, h - 50), (w - 30, h - 50), (0, 0, 255), 4)

    polygone = np.array([[10, h - 10], [50, h - 40], [90, h - 10]], np.int32)
    cv2.polylines(canvas, [polygone.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 255), thickness=3)

    return canvas


def add_text(img: np.ndarray, text: str) -> np.ndarray:
    canvas = img.copy()
    h, w = canvas.shape[:2]
    adaptive_scale = max(0.5, w / 900.0)
    cv2.putText(canvas, text, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, adaptive_scale, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


if __name__ == "__main__":
    if not os.path.exists(input_file):
        print("Image not found in file! Check if folder 'Day14' exists and has the image.")
    else:
        base_img = read_image(input_file)

        if base_img is not None:
            # 1. Grayscale Conversion
            gray = convert_grayscale(base_img)
            cv2.imwrite(os.path.join(output_dir, "task2_grayscale.jpg"), gray)

            # 2. Resize Images
            resized_dict = resize_multiple(base_img)
            for key, resized_image in resized_dict.items():
                filename = "task3_resize_" + key + ".jpg"
                cv2.imwrite(os.path.join(output_dir, filename), resized_image)

            # 3. Cropped Images
            cropped_dict = crop_regions(base_img)
            for key, cropped_image in cropped_dict.items():
                filename = "task4_crop_" + key + ".jpg"
                cv2.imwrite(os.path.join(output_dir, filename), cropped_image)

            # 4. Rotate Imahes
            rot_90 = rotate_img(base_img, degrees=90)
            rot_180 = rotate_img(base_img, degrees=180)
            rot_270 = rotate_img(base_img, degrees=270)
            cv2.imwrite(os.path.join(output_dir, "task5_rot_90.jpg"), rot_90)
            cv2.imwrite(os.path.join(output_dir, "task5_rot_180.jpg"), rot_180)
            cv2.imwrite(os.path.join(output_dir, "task5_rot_270.jpg"), rot_270)

            # 5. Flip Axis
            flip_h = flip_img(base_img, direction="Horizontal")
            flip_v = flip_img(base_img, direction="Vertical")
            cv2.imwrite(os.path.join(output_dir, "task6_flip_h.jpg"), flip_h)
            cv2.imwrite(os.path.join(output_dir, "task6_flip_v.jpg"), flip_v)

            # 6. Draw Shapes
            shapes = draw_shapes(base_img)
            cv2.imwrite(os.path.join(output_dir, "task7_shapes.jpg"), shapes)

            # 7. Aadd Name and Date
            annotate = add_text(shapes, text=my_name + " - " + date_tdy)
            cv2.imwrite(os.path.join(output_dir, "task8_annotate.jpg"), annotate)

            print("All operations successfully saved inside folder:", output_dir)
