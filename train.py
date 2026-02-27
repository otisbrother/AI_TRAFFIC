from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    model.train(
        data='data.yaml',
        epochs=3,      # ↓ giảm xuống 3 cho nhanh
        imgsz=416      # ↓ giảm size cho nhẹ máy
    )

if __name__ == '__main__':
    main()