import cv2
import numpy as np

def order_points(pts):
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered

def four_point_warp(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped

def detect_document_boundary(image, canny_low=None, canny_high=None,
                              blur_ksize=5, morph_ksize=5, resize_width=700):
    orig = image.copy()
    h, w = orig.shape[:2]
    if w > resize_width:
        scale = resize_width / w
        orig_resized = cv2.resize(orig, (resize_width, int(h * scale)))
    else:
        orig_resized = orig.copy()
    gray = cv2.cvtColor(orig_resized, cv2.COLOR_BGR2GRAY)
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    if canny_low is None or canny_high is None:
        median_val = np.median(blurred)
        canny_low = int(max(0, 0.66 * median_val))
        canny_high = int(min(255, 1.33 * median_val))
    edges = cv2.Canny(blurred, canny_low, canny_high)
    mk = morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk, mk))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:8]
    doc_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 0.15 * (orig_resized.shape[0] * orig_resized.shape[1]):
            doc_contour = approx
            break
    boundary_img = orig_resized.copy()
    warped = None
    found = doc_contour is not None
    if not found and contours:
        c = contours[0]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        doc_contour = np.intp(box).reshape(-1, 1, 2)
    if doc_contour is not None:
        cv2.drawContours(boundary_img, [doc_contour], -1, (0, 255, 0), 3)
        for point in doc_contour.reshape(-1, 2):
            cv2.circle(boundary_img, tuple(point), 6, (0, 0, 255), -1)
        if len(doc_contour) == 4:
            warped = four_point_warp(orig_resized, doc_contour.astype("float32"))
    return {
        "original": orig_resized, "gray": gray, "blurred": blurred,
        "edges": edges, "morphology": morph, "boundary": boundary_img,
        "warped": warped, "found_quad": found,
        "canny_thresholds": (canny_low, canny_high),
    }