import os
import csv
import glob
import cv2
import numpy as np
from boundary_detection_core import detect_document_boundary

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input_images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_images", "challenge_task")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def label(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def process(path, summary_rows):
    name = os.path.splitext(os.path.basename(path))[0]
    img = cv2.imread(path)
    if img is None:
        print(f"  [skip] could not read {path}")
        return

    result = detect_document_boundary(img)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_A_original.jpg"), result["original"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_B_edge_detection.jpg"), result["edges"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_C_morphological_result.jpg"), result["morphology"])
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_D_final_boundary.jpg"), result["boundary"])
    if result["warped"] is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_E_scanned_output.jpg"), result["warped"])

    tiles = [
        label(result["original"], "A: Original"),
        label(to_bgr(result["edges"]), "B: Edge Detection (Canny)"),
        label(to_bgr(result["morphology"]), "C: Morphological Result"),
        label(result["boundary"], "D: Final Boundary"),
    ]
    comparison = np.hstack(tiles)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_COMPARISON.jpg"), comparison)

    summary_rows.append({
        "image": os.path.basename(path),
        "canny_low": result["canny_thresholds"][0],
        "canny_high": result["canny_thresholds"][1],
        "quadrilateral_found": result["found_quad"],
        "scanned_output_saved": result["warped"] is not None,
    })
    print(f"  [ok] {name}  (quad_found={result['found_quad']})")

def main():
    images = sorted(
        glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.jpeg")) +
        glob.glob(os.path.join(INPUT_DIR, "*.png"))
    )
    if len(images) < 10:
        print(f"Warning: only {len(images)} images found (challenge asks for 10+).")
    print(f"Processing {len(images)} image(s) through the full pipeline...\n")

    summary_rows = []
    for path in images:
        process(path, summary_rows)

    csv_path = os.path.join(OUTPUT_DIR, "summary_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    found_count = sum(1 for r in summary_rows if r["quadrilateral_found"])
    print(f"\nDone. {found_count}/{len(summary_rows)} images had a clean 4-point boundary detected.")
    print(f"Results + summary CSV saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
