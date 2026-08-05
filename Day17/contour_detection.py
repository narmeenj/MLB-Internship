import os
import cv2
import numpy as np


def process_contours(input_folder="Day17/input_images", output_folder="Day17/output_images/output_contour"):
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        os.makedirs(input_folder, exist_ok=True)
        print(f"Created missing folder: '{input_folder}'.")

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    if not image_files:
        print("No images found in input folder. Creating a synthetic shapes image for demonstration...")
        synthetic_canvas = np.zeros((500, 500, 3), dtype=np.uint8)
        pts = np.array([[100, 100], [50, 200], [150, 200]], np.int32)
        cv2.drawContours(synthetic_canvas, [pts], 0, (255, 255, 255), -1)
        cv2.rectangle(synthetic_canvas, (240, 80), (420, 160), (255, 255, 255), -1)
        cv2.circle(synthetic_canvas, (120, 360), 55, (255, 255, 255), -1)
        cv2.rectangle(synthetic_canvas, (280, 280), (400, 400), (255, 255, 255), -1)

        fallback_name = "synthetic_demo.png"
        cv2.imwrite(os.path.join(input_folder, fallback_name), synthetic_canvas)
        image_files.append(fallback_name)

    print(f"Found {len(image_files)} image(s) to process.\n")

    for filename in image_files:
        image_path = os.path.join(input_folder, filename)
        img = cv2.imread(image_path)

        if img is None:
            print(f"Could not read {filename}. Skipping.")
            continue

        print(f"Processing image: {filename}")
        base_name = os.path.splitext(filename)[0]

        output_contour = img.copy()
        output_bbox = img.copy()
        output_shapes = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0 or (len(contours) == 1 and cv2.contourArea(contours[0]) > (img.shape[0] * img.shape[1] * 0.95)):
            thresh = thresh2
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            thresh = thresh1

        print(f"   Total initial contours discovered: {len(contours)}")

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            if area < 500 or area > (img.shape[0] * img.shape[1] * 0.95):
                continue

            print(f"   -> Contour #{i+1}: Area = {area:.2f} px | Perimeter = {perimeter:.2f} px")

            cv2.drawContours(output_contour, [cnt], -1, (0, 255, 0), 2)

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)

            epsilon = 0.04 * perimeter
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else x
            cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else y

            vertices = len(approx)
            if vertices == 3:
                shape_name = "Triangle"
            elif vertices == 4:
                aspect_ratio = float(w) / h
                shape_name = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif vertices == 5:
                shape_name = "Pentagon"
            elif vertices == 6:
                shape_name = "Hexagon"
            else:
                shape_name = "Circle"

            cv2.putText(
                output_shapes,
                shape_name,
                (cX - 25, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.drawContours(output_shapes, [approx], -1, (0, 255, 255), 2)

        cv2.imwrite(os.path.join(output_folder, f"{base_name}_1_grayscale.png"), gray)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_2_threshold.png"), thresh)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_3_contours.png"), output_contour)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_4_bounding_boxes.png"), output_bbox)
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_5_shape_detection.png"), output_shapes)
        print(f"Saved pipeline views for {filename} into '{output_folder}/'\n")

    print("All images processed and pipeline output states saved to file successfully!")


if __name__ == "__main__":
    process_contours()
