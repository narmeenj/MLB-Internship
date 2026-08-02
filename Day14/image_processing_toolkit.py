import os
import sys
from datetime import date

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import toolkit


MY_NAME = "Narmeen Javed"

MENU = """0
==================== IMAGE PROCESSING TOOLKIT ====================
 1. Load an image (shows height/width/channels/file size)
 2. Convert to grayscale
 3. Resize image (exact width/height)
 4. Resize to preset sizes (small/medium/large/wide) - saves all 4
 5. Rotate image (90 / 180 / 270)
 6. Rotate image (any angle)
 7. Flip image (horizontal / vertical / both)
 8. Crop image (custom coordinates)
 9. Crop into named regions (upper/lower/left/right/center/corners)
10. Draw demo shapes
11. Add custom text
12. Adjust brightness / contrast
13. Show original vs processed side by side
14. Run full original batch pipeline on a fresh image
15. Save current working image
16. Reset to original loaded image
 0. Exit
====================================================================
"""


def prompt_int(msg: str, default=None) -> int:
    raw = input(msg).strip()
    if raw == "" and default is not None:
        return default
    return int(raw)


def prompt_float(msg: str, default=None) -> float:
    raw = input(msg).strip()
    if raw == "" and default is not None:
        return default
    return float(raw)


def has_display() -> bool:
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY"))


def show(img, title="Image"):
    """Display an image in a window if a display is available."""
    if not has_display():
        print("(No display detected in this environment - skipping the "
              "preview window. The image was still processed; use option "
              "15 to save it instead.)")
        return
    try:
        cv2.imshow(title, img)
        print("Press any key in the image window to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("(Could not open a preview window - skipping. "
              "The image was still processed.)")


class App:
    def __init__(self):
        self.original = None
        self.working = None
        self.loaded_path = None

    def require_image(self):
        if self.working is None:
            raise ValueError("No image loaded yet. Choose option 1 first.")

    def load(self):
        path = input("Enter path to image file: ").strip()
        self.original = toolkit.read_image(path)   # prints the analysis
        self.working = self.original.copy()
        self.loaded_path = path

    def grayscale(self):
        self.require_image()
        self.working = toolkit.convert_grayscale(self.working)
        print("Converted to grayscale.")

    def resize_exact(self):
        self.require_image()
        w = prompt_int("New width (px): ")
        h = prompt_int("New height (px): ")
        self.working = toolkit.resize_img(self.working, w, h)
        print(f"Resized. New shape: {self.working.shape}")

    def resize_presets(self):
        self.require_image()
        out_dir = input("Folder to save the 4 preset resizes into: ").strip()
        os.makedirs(out_dir, exist_ok=True)
        resized = toolkit.resize_multiple(self.working)
        for name, img in resized.items():
            toolkit.save_image(img, os.path.join(out_dir, f"resize_{name}.jpg"))
        print(f"Saved {len(resized)} preset sizes to {out_dir}")

    def rotate_fixed(self):
        self.require_image()
        degrees = prompt_int("Rotate by 90, 180 or 270 degrees: ")
        self.working = toolkit.rotate_img(self.working, degrees)
        print(f"Rotated {degrees} degrees.")

    def rotate_free(self):
        self.require_image()
        angle = prompt_float("Rotation angle in degrees (any value): ")
        self.working = toolkit.rotate_any(self.working, angle)
        print(f"Rotated by {angle} degrees (canvas expanded to avoid clipping).")

    def flip(self):
        self.require_image()
        mode = input("Flip mode - horizontal / vertical / both: ").strip().lower()
        self.working = toolkit.flip_img(self.working, mode)
        print(f"Flipped ({mode}).")

    def crop_custom(self):
        self.require_image()
        h, w = self.working.shape[:2]
        print(f"Current image size: width={w}, height={h}")
        x = prompt_int("Crop top-left x: ")
        y = prompt_int("Crop top-left y: ")
        cw = prompt_int("Crop width: ")
        ch = prompt_int("Crop height: ")
        self.working = toolkit.crop_image(self.working, x, y, cw, ch)
        print(f"Cropped. New shape: {self.working.shape}")

    def crop_named_regions(self):
        self.require_image()
        out_dir = input("Folder to save the named-region crops into: ").strip()
        os.makedirs(out_dir, exist_ok=True)
        regions = toolkit.crop_regions(self.working)
        for name, img in regions.items():
            toolkit.save_image(img, os.path.join(out_dir, f"crop_{name}.jpg"))
        print(f"Saved {len(regions)} region crops to {out_dir}")

    def draw_shapes(self):
        self.require_image()
        self.working = toolkit.draw_shapes(self.working)
        print("Drew demo shapes (rectangle, circle, line, triangle).")

    def add_text(self):
        self.require_image()
        default_text = f"{MY_NAME} - {date.today().strftime('%d-%m-%Y')}"
        text = input(f"Text to add [default: '{default_text}']: ").strip()
        if not text:
            text = default_text
        self.working = toolkit.add_text(self.working, text)
        print("Added text.")

    def brightness_contrast(self):
        self.require_image()
        b = prompt_int("Brightness (-100 to 100) [0]: ", default=0)
        c = prompt_int("Contrast (-100 to 100) [0]: ", default=0)
        self.working = toolkit.adjust_brightness_contrast(self.working, b, c)
        print("Adjusted brightness/contrast.")

    def compare_side_by_side(self):
        self.require_image()
        if self.original is None:
            raise ValueError("No original image stored to compare against.")
        combined = toolkit.side_by_side(self.original, self.working)
        show(combined, "Left: Original | Right: Processed")

    def run_batch_pipeline(self):
        """Reproduces the original script's full pipeline on one image."""
        path = input("Enter path to image file: ").strip()
        out_dir = input("Output folder for batch results: ").strip()
        os.makedirs(out_dir, exist_ok=True)

        img = toolkit.read_image(path)

        gray = toolkit.convert_grayscale(img)
        toolkit.save_image(gray, os.path.join(out_dir, "task2_grayscale.jpg"))

        for name, resized in toolkit.resize_multiple(img).items():
            toolkit.save_image(resized, os.path.join(out_dir, f"task3_resize_{name}.jpg"))

        for name, cropped in toolkit.crop_regions(img).items():
            toolkit.save_image(cropped, os.path.join(out_dir, f"task4_crop_{name}.jpg"))

        for degrees in (90, 180, 270):
            rotated = toolkit.rotate_img(img, degrees)
            toolkit.save_image(rotated, os.path.join(out_dir, f"task5_rot_{degrees}.jpg"))

        toolkit.save_image(toolkit.flip_img(img, "horizontal"),
                            os.path.join(out_dir, "task6_flip_h.jpg"))
        toolkit.save_image(toolkit.flip_img(img, "vertical"),
                            os.path.join(out_dir, "task6_flip_v.jpg"))

        shapes = toolkit.draw_shapes(img)
        toolkit.save_image(shapes, os.path.join(out_dir, "task7_shapes.jpg"))

        annotated = toolkit.add_text(
            shapes, text=f"{MY_NAME} - {date.today().strftime('%d-%m-%Y')}")
        toolkit.save_image(annotated, os.path.join(out_dir, "task8_annotate.jpg"))

        print("All batch operations successfully saved inside folder:", out_dir)

    def save(self):
        self.require_image()
        path = input("Enter output path (e.g. output.jpg): ").strip()
        toolkit.save_image(self.working, path)
        print(f"Saved to {path}")

    def reset(self):
        self.require_image()
        self.working = self.original.copy()
        print("Working image reset to the original loaded image.")


def main():
    app = App()
    actions = {
        "1": app.load,
        "2": app.grayscale,
        "3": app.resize_exact,
        "4": app.resize_presets,
        "5": app.rotate_fixed,
        "6": app.rotate_free,
        "7": app.flip,
        "8": app.crop_custom,
        "9": app.crop_named_regions,
        "10": app.draw_shapes,
        "11": app.add_text,
        "12": app.brightness_contrast,
        "13": app.compare_side_by_side,
        "14": app.run_batch_pipeline,
        "15": app.save,
        "16": app.reset,
    }

    print("Welcome to the Image Processing Toolkit!")
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()
        if choice == "0":
            print("Goodbye!")
            break
        action = actions.get(choice)
        if action is None:
            print("Invalid option, please try again.")
            continue
        try:
            action()
        except ValueError as e:
            print(f"[Error] {e}")
        except Exception as e: 
            print(f"[Unexpected error] {e}")


if __name__ == "__main__":
    main()
