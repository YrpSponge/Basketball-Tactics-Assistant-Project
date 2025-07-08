from ultralytics import YOLO
import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = YOLO("models/v8l_detection_training.pt")
model.track("sources/game_clip1.mp4", show=True)
    