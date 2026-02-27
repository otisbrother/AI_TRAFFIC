# Predict traffic objects using trained YOLOv8 model
from ultralytics import YOLO
import cv2
import sys

def main():
    model = YOLO('runs/detect/train2/weights/best.pt')  # Update path if needed
    img_path = sys.argv[1] if len(sys.argv) > 1 else 'test.jpg'
    results = model.predict(img_path, show=True)
    # Optionally save or process results here

if __name__ == '__main__':
    main()



