import cv2
import numpy as np
import os
import sys


def translate_image(img, tx, ty):
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx],
                     [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def rotate_image(img, angle, scale=1.0, center=None):
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, M, (w, h))


def scale_image(img, fx, fy, interpolation=None):
    if interpolation is None:
        interpolation = cv2.INTER_CUBIC if fx > 1 or fy > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)


def affine_transform(img, src_pts=None, dst_pts=None):
    h, w = img.shape[:2]
    if src_pts is None:
        src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
    if dst_pts is None:
        dst_pts = np.float32([[0, 0], [int(w * 0.9), int(h * 0.1)], [int(w * 0.1), h - 1]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    return cv2.warpAffine(img, M, (w, h))


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_transform(img, pts):
    rect = order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def auto_detect_document_corners(img):
    """
    Try to automatically find the 4 corners of a document in the image
    using edge detection + contour analysis. Returns None if not found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * img.shape[0] * img.shape[1]:
            return approx.reshape(4, 2)
    return None


def _make_demo_image(path):
    """Create a synthetic 'document' test image if none is supplied."""
    canvas = np.full((600, 800, 3), 40, dtype=np.uint8)
    doc = np.full((400, 550, 3), 255, dtype=np.uint8)
    for i in range(6):
        cv2.line(doc, (30, 40 + i * 55), (520, 40 + i * 55), (60, 60, 60), 3)
    cv2.putText(doc, "SAMPLE DOCUMENT", (60, 380), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 0, 0), 2)
    h, w = doc.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[130, 60], [740, 20], [700, 560], [90, 520]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_doc = cv2.warpPerspective(doc, M, (800, 600), borderValue=(40, 40, 40))
    canvas = np.where(warped_doc.sum(axis=2, keepdims=True) > 0, warped_doc, canvas)
    cv2.imwrite(path, canvas)
    return canvas


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "output_images", "transformations")
    os.makedirs(out_dir, exist_ok=True)

    input_dir = os.path.join(os.path.dirname(__file__), "input_images")
    own_images = []
    if os.path.isdir(input_dir):
        own_images = [f for f in os.listdir(input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        image = cv2.imread(sys.argv[1])
        print(f"Using image passed on command line: {sys.argv[1]}")
    elif own_images:
        chosen = os.path.join(input_dir, own_images[0])
        image = cv2.imread(chosen)
        print(f"Using your own image from input_images: {chosen}")
    else:
        demo_path = os.path.join(out_dir, "demo_source.jpg")
        image = _make_demo_image(demo_path)
        print(f"No image found in input_images -> generated a demo tilted document at {demo_path}")

    cv2.imwrite(os.path.join(out_dir, "01_original.jpg"), image)
    cv2.imwrite(os.path.join(out_dir, "02_translated.jpg"), translate_image(image, 60, 40))
    cv2.imwrite(os.path.join(out_dir, "03_rotated_30.jpg"), rotate_image(image, 30))
    cv2.imwrite(os.path.join(out_dir, "04_rotated_-45.jpg"), rotate_image(image, -45))
    cv2.imwrite(os.path.join(out_dir, "05_scaled_up.jpg"), scale_image(image, 1.5, 1.5))
    cv2.imwrite(os.path.join(out_dir, "06_scaled_down.jpg"), scale_image(image, 0.5, 0.5))
    cv2.imwrite(os.path.join(out_dir, "07_affine.jpg"), affine_transform(image))

    corners = auto_detect_document_corners(image)
    if corners is not None:
        warped = perspective_transform(image, corners)
        cv2.imwrite(os.path.join(out_dir, "08_perspective_corrected.jpg"), warped)
        print("Perspective correction applied using auto-detected corners:", corners.tolist())
    else:
        print("Could not auto-detect 4 document corners; supply points manually via perspective_transform().")

    print(f"All transformation examples saved to: {out_dir}")