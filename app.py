"""Streamlit web demo for traffic-object detection."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st

from traffic_utils import DEFAULT_MODEL, load_yolo

st.set_page_config(page_title="AI Traffic Detection", page_icon="🚦", layout="wide")
st.title("🚦 Nhận diện đối tượng giao thông")
st.caption("YOLOv8 · car · motorbike · bicycle · person · truck")


@st.cache_resource
def cached_model(model_path: str):
    return load_yolo(model_path)


with st.sidebar:
    st.header("Cấu hình")
    model_path = st.text_input("Model", value=str(DEFAULT_MODEL))
    confidence = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, 0.70, 0.05)

uploaded = st.file_uploader("Chọn ảnh giao thông", type=["jpg", "jpeg", "png", "webp"])
if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(uploaded.getbuffer())
        source_path = temporary.name
    try:
        with st.spinner("Đang nhận diện..."):
            result = cached_model(model_path).predict(
                source=source_path, conf=confidence, iou=iou, verbose=False
            )[0]
        left, right = st.columns(2)
        left.image(uploaded, caption="Ảnh gốc", use_container_width=True)
        right.image(result.plot()[:, :, ::-1], caption="Kết quả", use_container_width=True)
        counts = Counter(result.names[int(class_id)] for class_id in result.boxes.cls.tolist())
        st.subheader(f"Phát hiện {sum(counts.values())} đối tượng")
        if counts:
            st.dataframe(
                {"Đối tượng": list(counts.keys()), "Số lượng": list(counts.values())},
                use_container_width=True,
                hide_index=True,
            )
    except (FileNotFoundError, RuntimeError) as error:
        st.error(str(error))
    finally:
        Path(source_path).unlink(missing_ok=True)
else:
    st.info("Tải một ảnh lên để bắt đầu nhận diện.")
