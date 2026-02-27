# Evaluate YOLOv8 model for traffic object detection
from ultralytics import YOLO

def main():
    model = YOLO('runs/detect/train2/weights/best.pt')  # Update path if needed
    metrics = model.val()
    print(metrics)

if __name__ == '__main__':
    main()
