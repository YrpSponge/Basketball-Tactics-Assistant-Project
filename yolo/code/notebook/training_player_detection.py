from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    model = YOLO("../models/yolov8l.pt")  # 使用轻量模型 yolov8n (nano)，更省显存
    model.train(
        data="dataset/Basketball-Players-25/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        cache=False,
        device=0,
        amp=False
    )

if __name__ == "__main__":
    main()