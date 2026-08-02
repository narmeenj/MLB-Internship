from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results_single = model("https://ultralytics.com", save=True)
source_list = [
    "https://ultralytics.com",
    "https://ultralytics.com"
]
results_multiple = model(source_list, save=True)

print("Inference complete! Check the 'runs/detect/' directory for your saved images.")
