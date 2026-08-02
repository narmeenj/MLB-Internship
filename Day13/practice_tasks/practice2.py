my_image_path = "my_photo.jpg" 

results = model(my_image_path, save=True)
for result in results:
    boxes = result.boxes
    for box in boxes:
        class_id = int(box.cls[0])
        score = float(box.conf[0])
        coordinates = box.xyxy[0].tolist()
        
        print(f"Detected Class ID: {class_id} | Confidence: {score:.2f} | Box: {coordinates}")
