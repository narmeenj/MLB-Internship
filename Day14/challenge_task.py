import os
import sys
from datetime import date

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "image_toolkit"))
import toolkit  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
OUT_DIR = os.path.join(BASE_DIR, "processed_outputs")

MY_NAME = "Narmeen Javed"

CATEGORIES = {
    "landscape": "landscape.jpg",
    "person": "person.jpg",
    "vehicle": "vehicle.jpg",
    "document": "document.jpg",
    "object": "object.jpg",
}


def process_one(category: str, filename: str):
    in_path = os.path.join(SAMPLE_DIR, filename)
    out_dir = os.path.join(OUT_DIR, category)
    os.makedirs(out_dir, exist_ok=True)

    img = toolkit.read_image(in_path, verbose=False)

    gray = toolkit.convert_grayscale(img)
    toolkit.save_image(gray, os.path.join(out_dir, "01_grayscale.jpg"))

    for name, resized in toolkit.resize_multiple(img).items():
        toolkit.save_image(resized, os.path.join(out_dir, f"02_resize_{name}.jpg"))

    for name, cropped in toolkit.crop_regions(img).items():
        toolkit.save_image(cropped, os.path.join(out_dir, f"03_crop_{name}.jpg"))

    for degrees in (90, 180, 270):
        rotated = toolkit.rotate_img(img, degrees)
        toolkit.save_image(rotated, os.path.join(out_dir, f"04_rotate_{degrees}.jpg"))

    rotated_45 = toolkit.rotate_any(img, 45)
    toolkit.save_image(rotated_45, os.path.join(out_dir, "05_rotate_45_free_angle.jpg"))

    toolkit.save_image(toolkit.flip_img(img, "horizontal"),
                        os.path.join(out_dir, "06_flip_horizontal.jpg"))
    toolkit.save_image(toolkit.flip_img(img, "vertical"),
                        os.path.join(out_dir, "06_flip_vertical.jpg"))

    shapes = toolkit.draw_shapes(img)
    toolkit.save_image(shapes, os.path.join(out_dir, "07_shapes.jpg"))

    annotated = toolkit.add_text(
        shapes, text=f"{category} - {MY_NAME} - {date.today().strftime('%d-%m-%Y')}")
    toolkit.save_image(annotated, os.path.join(out_dir, "08_text.jpg"))

    brighter = toolkit.adjust_brightness_contrast(img, brightness=40, contrast=0)
    toolkit.save_image(brighter, os.path.join(out_dir, "09_brightness_up.jpg"))

    higher_contrast = toolkit.adjust_brightness_contrast(img, brightness=0, contrast=40)
    toolkit.save_image(higher_contrast, os.path.join(out_dir, "10_contrast_up.jpg"))

    comparison = toolkit.side_by_side(img, gray)
    toolkit.save_image(comparison,
                        os.path.join(out_dir, "11_side_by_side_original_vs_gray.jpg"))

    total_files = 1 + 4 + 9 + 3 + 1 + 2 + 1 + 1 + 1 + 1 + 1
    print(f"[{category}] processed {total_files} outputs -> {out_dir}")


def main():
    for category, filename in CATEGORIES.items():
        process_one(category, filename)
    print("\nChallenge task complete. See the processed_outputs/ folder.")


if __name__ == "__main__":
    main()
