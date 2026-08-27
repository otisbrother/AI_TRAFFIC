<div align="center">

# 🚦 AI TRAFFIC DETECTION

### Hệ thống nhận diện và phân tích đối tượng giao thông với YOLOv8

*Phát hiện phương tiện và người tham gia giao thông từ ảnh, video, webcam hoặc camera IP — từ kiểm tra dữ liệu đến huấn luyện, đánh giá và triển khai demo.*

[Mã nguồn](https://github.com/otisbrother/AI_TRAFFIC) · [Xem kết quả mẫu](#-demo) · [Khởi chạy nhanh](#-khởi-chạy-nhanh) · [Kết quả đánh giá](#-kết-quả-mô-hình)

</div>

---

## Tổng quan dành cho nhà tuyển dụng

AI Traffic Detection là một dự án Computer Vision hoàn chỉnh, giải quyết bài toán phát hiện đối tượng trong bối cảnh giao thông đô thị. Hệ thống không chỉ dừng ở notebook thử nghiệm mà được tổ chức thành một pipeline có thể tái lập: **audit dữ liệu → huấn luyện → đánh giá → suy luận → trực quan hóa → kiểm thử tự động**.

| Hạng mục | Kết quả hiện tại |
| --- | ---: |
| Dữ liệu | **6.354 ảnh** |
| Bounding box | **76.440 đối tượng** |
| Lớp nhận diện | **5 lớp** |
| Baseline mAP@50 | **71,3%** |
| Baseline Precision | **78,8%** |
| Kiểm tra định dạng nhãn | **0 lỗi / 6.354 ảnh** |
| Unit test | **3/3 passed** |

> **Mức độ hoàn thiện:** functional prototype / engineering portfolio. Pipeline đã sẵn sàng để train lại và mở rộng; checkpoint hiện có là baseline nghiên cứu, chưa phải model production.

### Giá trị có thể ứng dụng

- Giám sát mật độ phương tiện tại nút giao hoặc tuyến đường.
- Làm đầu vào cho bài toán đếm xe, theo dõi luồng giao thông và cảnh báo ùn tắc.
- Hỗ trợ dashboard vận hành đô thị thông minh từ camera hiện hữu.
- Mở rộng sang edge AI, camera IP, tracking và phân tích theo vùng quan tâm.

---

## Bài toán và giải pháp

Camera giao thông tạo ra lượng dữ liệu lớn nhưng việc quan sát thủ công khó mở rộng và không cung cấp dữ liệu định lượng theo thời gian thực. Dự án sử dụng **YOLOv8n** để chuyển dữ liệu hình ảnh thành các detection có cấu trúc:

```text
Ảnh / Video / Camera
        ↓
Bounding box + loại đối tượng + độ tin cậy
        ↓
Media đã gán nhãn + thống kê JSON
```

Hệ thống nhận diện năm nhóm đối tượng:

| ID | Nhãn | Ý nghĩa |
| ---: | --- | --- |
| 0 | `car` | Ô tô con |
| 1 | `motorbike` | Xe máy |
| 2 | `bicycle` | Xe đạp |
| 3 | `person` | Người |
| 4 | `truck` | Xe tải |

---

## Demo

Web demo được cung cấp bằng Streamlit và chạy cục bộ:

```bash
streamlit run app.py
```

Giao diện cho phép tải ảnh, điều chỉnh ngưỡng confidence/IoU, so sánh ảnh trước–sau và xem số lượng đối tượng theo lớp. Dự án hiện **chưa triển khai một URL demo công khai**.

---

## Kiến trúc hệ thống

```mermaid
flowchart TB
    subgraph Data[Data pipeline]
        A[YOLO Dataset] --> B[Dataset Audit]
        B -->|valid| C[Train / Validation]
        B -->|invalid| D[Error Report]
    end

    subgraph Model[Model pipeline]
        C --> E[YOLOv8 Checkpoint]
        E --> F[Evaluation]
        F --> G[Metrics JSON + Plots]
    end

    subgraph Inference[Inference pipeline]
        H[Image / Video / Webcam / RTSP] --> I[Pre-processing]
        I --> E
        E --> J[Detection Results]
        J --> K[Annotated Media]
        J --> L[Class Statistics JSON]
        J --> M[Streamlit UI]
    end
```

### Luồng kỹ thuật

1. `inspect_dataset.py` kiểm tra tính toàn vẹn của ảnh và nhãn YOLO.
2. `train.py` fine-tune YOLOv8 với cấu hình có thể thay đổi qua CLI.
3. `evaluate.py` đánh giá checkpoint và xuất metrics dạng máy đọc được.
4. `predict.py` suy luận theo cơ chế streaming để không giữ toàn bộ video trong RAM.
5. `app.py` cung cấp giao diện demo cho người dùng không chuyên kỹ thuật.
6. GitHub Actions chạy syntax check, unit test và dataset audit trên mỗi push/PR.

---

## Chức năng chính

### Data quality

- Đối chiếu ảnh và tệp nhãn theo tên.
- Phát hiện ảnh thiếu nhãn hoặc nhãn không có ảnh.
- Kiểm tra số cột, class ID, tọa độ chuẩn hóa và kích thước bounding box.
- Thống kê số ảnh, số đối tượng, nhãn rỗng và phân bố từng lớp.
- Chế độ `--strict` trả exit code khác 0 để tích hợp CI/CD.

### Training và evaluation

- Cấu hình model, epochs, image size, batch size, device và worker từ CLI.
- Early stopping, resume training, seed cố định và deterministic mode.
- Hỗ trợ CPU hoặc GPU mà không cần sửa mã nguồn.
- Xuất Precision, Recall, mAP và tốc độ xử lý thành JSON.

### Inference và demo

- Đầu vào: ảnh, thư mục ảnh, video, webcam, HTTP/RTSP/RTMP stream.
- Cấu hình confidence, IoU, image size và device.
- Lưu media đã gán nhãn, YOLO text labels tùy chọn và `summary.json`.
- Xử lý video/stream theo generator nhằm giảm áp lực bộ nhớ.
- Tự động ẩn username/password khỏi URL camera trong log và báo cáo.
- Web UI hỗ trợ upload ảnh và hiển thị thống kê trực tiếp.

---

## Dữ liệu và kiểm soát chất lượng

### Quy mô dataset hiện tại

| Split | Số ảnh | Số đối tượng | Ảnh nền / nhãn rỗng | Lỗi định dạng |
| --- | ---: | ---: | ---: | ---: |
| Train | 5.805 | 70.170 | 93 | 0 |
| Validation | 549 | 6.270 | 8 | 0 |
| **Tổng** | **6.354** | **76.440** | **101** | **0** |

### Phân bố lớp

| Lớp | Train | Validation | Tổng | Tỷ lệ |
| --- | ---: | ---: | ---: | ---: |
| car | 2.766 | 250 | 3.016 | 3,9% |
| motorbike | 1.032 | 108 | 1.140 | 1,5% |
| bicycle | 41.478 | 3.842 | 45.320 | 59,3% |
| person | 13.995 | 1.238 | 15.233 | 19,9% |
| truck | 10.899 | 832 | 11.731 | 15,4% |

Dataset hợp lệ về mặt định dạng nhưng **mất cân bằng đáng kể**: `bicycle` chiếm 59,3%, trong khi `motorbike` chỉ chiếm 1,5%. Đây là rủi ro làm model thiên lệch và cần được xử lý bằng thu thập bổ sung, sampling, augmentation hoặc class-aware training.

Chạy lại báo cáo bất kỳ lúc nào:

```bash
python inspect_dataset.py --strict
```

Báo cáo chi tiết được ghi tại `runs/dataset-report.json`.

---

## Kết quả mô hình

### Baseline trên validation set

| Metric | Giá trị |
| --- | ---: |
| Precision | **0,788** |
| Recall | **0,641** |
| mAP@50 | **0,713** |
| mAP@50–95 | **0,482** |

### Diễn giải trung thực kết quả

Checkpoint hiện có được huấn luyện bằng YOLOv8n trên **CPU**, image size **416 px**. Cấu hình đặt 3 epochs nhưng `results.csv` chỉ ghi nhận epoch đầu tiên, vì vậy các metric trên được xem là **baseline**, không phải kết quả hội tụ cuối cùng.

Ngoài ra, artifact của lần train cũ ghi nhận khoảng **32.694 đối tượng và chưa có mẫu `truck`**, trong khi dataset hiện tại có 76.440 đối tượng, bao gồm 11.731 `truck`. Vì vậy:

- Không sử dụng checkpoint cũ để kết luận hiệu quả nhận diện `truck`.
- Cần train lại trên snapshot dữ liệu hiện tại trước khi so sánh model.
- Cần bổ sung test set độc lập trước khi báo cáo khả năng tổng quát hóa.

Pipeline cải tiến mặc định dùng 30 epochs, image size 640, early stopping 10 epochs và seed 42. Cấu hình này là điểm khởi đầu; hyperparameter cuối cùng cần được chọn qua thực nghiệm có kiểm soát.

---

## Khởi chạy nhanh

### 1. Yêu cầu môi trường

- Python 3.10 trở lên.
- Khuyến nghị GPU NVIDIA/CUDA khi huấn luyện.
- CPU vẫn phù hợp cho kiểm thử pipeline và suy luận quy mô nhỏ.

### 2. Cài đặt

```bash
git clone https://github.com/otisbrother/AI_TRAFFIC.git
cd AI_TRAFFIC
python -m venv .venv
```

Kích hoạt môi trường trên Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Trên Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Xác minh dataset

```bash
python inspect_dataset.py --strict
```

### 4. Chạy web demo

```bash
streamlit run app.py
```

---

## Hướng dẫn sử dụng

### Huấn luyện trên GPU

```bash
python train.py \
  --model yolov8n.pt \
  --epochs 30 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

### Huấn luyện thử trên CPU

```bash
python train.py --epochs 10 --imgsz 416 --batch 8 --device cpu
```

### Đánh giá checkpoint

```bash
python evaluate.py \
  --model runs/detect/traffic-yolov8n/weights/best.pt \
  --imgsz 640 \
  --device 0
```

Metrics được lưu mặc định tại `runs/evaluation/metrics.json`.

### Nhận diện ảnh

```bash
python predict.py path/to/image.jpg --conf 0.30 --name image-demo
```

### Nhận diện video

```bash
python predict.py path/to/video.mp4 --name video-demo
```

### Webcam

```bash
python predict.py 0 --show
```

### Camera IP / RTSP

```bash
python predict.py rtsp://user:password@camera-host/stream --show
```

> Thông tin đăng nhập trong RTSP URL được loại khỏi `summary.json` và nội dung log do ứng dụng tạo ra.

### Các tùy chọn quan trọng

```bash
python train.py --help
python evaluate.py --help
python predict.py --help
python inspect_dataset.py --help
```

---

## Đầu ra suy luận

Mỗi lần chạy tạo một thư mục trong `runs/detect/<name>/`:

```text
runs/detect/image-demo/
├── image.jpg          # Ảnh đã vẽ bounding box
├── labels/            # Có khi dùng --save-txt
└── summary.json       # Thống kê có cấu trúc
```

Ví dụ `summary.json`:

```json
{
  "frames": 1,
  "detections": 8,
  "by_class": {
    "car": 3,
    "motorbike": 2,
    "person": 3
  },
  "mean_confidence": 0.7642,
  "source": "path/to/image.jpg",
  "output_dir": "runs/detect/image-demo"
}
```

---

## Cấu trúc repository

```text
AI_TRAFFIC/
├── .github/workflows/
│   └── quality.yml          # CI: syntax, test, dataset audit
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
├── runs/
│   └── detect/              # Checkpoint, biểu đồ và demo output
├── tests/
│   └── test_core.py         # Unit tests cho logic cốt lõi
├── app.py                   # Streamlit web application
├── data.yaml                # Dataset schema và class mapping
├── evaluate.py              # Model evaluation + JSON export
├── inspect_dataset.py       # YOLO dataset quality gate
├── predict.py               # Multi-source streaming inference
├── traffic_utils.py         # Shared validation, logging và summary
├── train.py                 # Configurable training pipeline
└── requirements.txt         # Python dependencies
```

---

## Chất lượng phần mềm

Dự án tách logic dùng chung khỏi các entry point, tránh hard-code đường dẫn và cho phép cấu hình hoàn toàn qua CLI. Các điểm kiểm soát kỹ thuật gồm:

- Lazy-load Ultralytics để các công cụ audit/test không phụ thuộc toàn bộ ML stack.
- Kiểm tra file đầu vào và hiển thị lỗi có ngữ cảnh.
- Validate khoảng giá trị của confidence và IoU.
- JSON output dùng UTF-8 và tương thích Windows.
- Không đưa cache dataset phụ thuộc máy cá nhân vào Git.
- Redact credential của camera stream trước khi ghi báo cáo.
- CI chạy tự động trên push và pull request.

Chạy kiểm thử cục bộ:

```bash
python -m py_compile traffic_utils.py train.py evaluate.py predict.py inspect_dataset.py app.py
python -m unittest discover -s tests -v
python inspect_dataset.py --strict
```

Trạng thái kiểm tra gần nhất:

```text
Unit tests:       3/3 passed
Dataset audit:    6.354 images, 76.440 objects, 0 format errors
Syntax check:     passed
```

---

## Giới hạn hiện tại

- Chưa có test set độc lập; validation metrics chưa đủ để kết luận production readiness.
- Dataset mất cân bằng lớp và nguồn dữ liệu chưa được mô tả đầy đủ về license/data lineage.
- Checkpoint cũ chưa đại diện cho dataset hiện tại và chưa học lớp `truck`.
- Chưa đo benchmark latency, throughput, tài nguyên CPU/GPU hoặc hiệu năng theo từng thiết bị.
- Chưa có tracking ID, đếm xe theo line/ROI hoặc phân tích chuỗi thời gian.
- Web demo hiện chạy local, chưa có authentication, monitoring và deployment manifest.

Việc công khai các giới hạn này là chủ đích: một hệ thống AI đáng tin cậy cần tách biệt rõ **prototype metrics** với **production evidence**.

---

## Lộ trình phát triển

### Giai đoạn 1 — Model quality

- Chuẩn hóa nguồn dữ liệu, giấy phép và version dataset.
- Bổ sung test set độc lập, đặc biệt cho `motorbike`, `car` và trường hợp khó.
- Train lại YOLOv8n/YOLOv8s trên GPU; theo dõi experiment và per-class metrics.
- Xử lý mất cân bằng lớp, phân tích confusion matrix và tối ưu threshold.

### Giai đoạn 2 — Traffic analytics

- Tích hợp ByteTrack/BoT-SORT để theo dõi đối tượng qua nhiều frame.
- Đếm phương tiện qua virtual line và vùng ROI.
- Ước lượng mật độ, lưu lượng và cảnh báo bất thường.
- Dashboard biểu đồ theo camera và thời gian.

### Giai đoạn 3 — Productionization

- Đóng gói Docker và cung cấp REST API bằng FastAPI.
- Export ONNX/TensorRT, benchmark trên GPU và thiết bị edge.
- Thêm authentication, structured logging, health check và monitoring.
- Data/model versioning, model registry, drift detection và quy trình rollback.

---

## Năng lực kỹ thuật thể hiện qua dự án

- Hiểu và triển khai end-to-end Object Detection pipeline.
- Làm việc với YOLO annotation, class mapping và data quality validation.
- Fine-tune, đánh giá và diễn giải Precision, Recall, mAP, confusion matrix.
- Thiết kế CLI, structured output và streaming inference.
- Xây dựng web demo phục vụ stakeholder không chuyên ML.
- Áp dụng unit testing, CI, security hygiene và reproducibility.
- Nhận diện đúng giới hạn dữ liệu/model và xây dựng roadmap hướng production.

---

## Thông tin dự án

| Thuộc tính | Nội dung |
| --- | --- |
| Tên | AI Traffic Detection |
| Lĩnh vực | Computer Vision · Intelligent Transportation System |
| Công nghệ | Python · YOLOv8 · PyTorch · OpenCV · Streamlit |
| Thời gian | 27/02/2026 – 27/08/2026 |
| Repository | [github.com/otisbrother/AI_TRAFFIC](https://github.com/otisbrother/AI_TRAFFIC) |
| Tác giả | [@otisbrother](https://github.com/otisbrother) |

<div align="center">

**From pixels to actionable traffic data.**

</div>
