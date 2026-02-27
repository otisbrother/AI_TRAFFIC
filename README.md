# Traffic Object Detection using YOLOv8

## Mục tiêu
Nhận diện ô tô, xe máy, xe đạp, người đi bộ từ hình ảnh giao thông.

## Công nghệ
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- LabelImg
- Matplotlib
- Kaggle Dataset

## Cấu trúc thư mục
```
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
data.yaml
train.py
predict.py
evaluate.py
README.md
```

## Hướng dẫn sử dụng
1. **Cài đặt phụ thuộc:**
   ```bash
   pip install ultralytics opencv-python matplotlib labelImg
   ```
2. **Tải và gán nhãn dataset:**
   - Dùng LabelImg để gán nhãn các class: car, motorbike, bicycle, person.
   - Xuất file nhãn dạng YOLO vào thư mục `labels/`.
3. **Train model:**
   ```bash
   python train.py
   ```
4. **Đánh giá:**
   ```bash
   python evaluate.py
   ```
5. **Dự đoán:**
   ```bash
   python predict.py test.jpg
   ```

## Nâng cấp
- So sánh YOLOv8n vs YOLOv8s
- Tính FPS
- Web demo bằng Flask
- Vẽ biểu đồ loss bằng matplotlib
