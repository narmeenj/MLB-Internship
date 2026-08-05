import cv2
import numpy as np


class ShapeDetector:

    def __init__(self, min_area=200):
        self.min_area = min_area

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return gray, thresh

    def detect_shape(self, contour, perimeter):
        area = cv2.contourArea(contour)
        
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter**2)
        else:
            circularity = 0

        if circularity > 0.82:
            return "Circle", contour

        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        shape = "Unknown Shape"

        if vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            extent = float(area) / rect_area

            if extent > 0.85:
                aspect_ratio = float(w) / float(h)
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            else:
                shape = "Polygon"
        elif vertices >= 5:
            shape = "Polygon"

        return shape, approx

    def process_image(self, image):
        if image is None:
            raise ValueError("Invalid image input.")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        gray, thresh = self.preprocess(image)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        contour_img = image.copy()
        final_img = image.copy()

        detected_stats = []

        for idx, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            perimeter = cv2.arcLength(c, True)
            shape, draw_geometry = self.detect_shape(c, perimeter)

            cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 2)

            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(c)
                cX, cY = x + w // 2, y + h // 2

            cv2.drawContours(final_img, [draw_geometry], -1, (255, 0, 0), 2)
            cv2.circle(final_img, (cX, cY), 4, (0, 0, 255), -1)

            label = f"{shape}"
            cv2.putText(
                final_img,
                label,
                (cX - 30, cY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )

            detected_stats.append({
                "Shape #": idx + 1,
                "Type": shape,
                "Area (px²)": round(area, 2),
                "Perimeter (px)": round(perimeter, 2),
                "Vertices": len(draw_geometry),
            })

        return contour_img, final_img, detected_stats


if __name__ == "__main__":
    print("ShapeDetector engine setup complete and ready for import.")
