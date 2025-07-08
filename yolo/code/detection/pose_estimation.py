from ultralytics import YOLO
import cv2
import torch
import os
import json
import numpy as np

class PoseEstimator:
    def __init__(self, model_path:str):
        self.model = YOLO(model_path)
        self.model.to('cuda')

        # # video input
        # self.video_path = video_path
        # self.cap = cv2.VideoCapture(video_path)

        # # create output dir
        # self.output_dir = output_dir
        # os.makedirs(self.output_dir, exist_ok=True)

        # # results saving
        # self.pose_dict = {}
        # self.frame_idx = 0

    def estimate(self, frame):
        # get the pose estimation results
        results = self.model(frame)
        poses = []
        
        for kp in results[0].keypoints:
            coords = kp.xy.cpu().numpy()
            confs = kp.conf.cpu().numpy()
            poses.append({
                "keypoints": coords[0].tolist(),
                "confs": confs[0].tolist(),
                "center": coords[0].mean(axis=0).tolist()
            })

        return poses
                


# --------old version----------
#     def estimate(self):
#         while self.cap.isOpened():
#             sucess, frame = self.cap.read()
#             if not sucess:
#                 break

#             results = self.model(frame)
#             keypoints = results[0].keypoints # all keypoints of people in a frame

#             for kp in keypoints:
#                 coords = kp.xy.cpu().numpy()
#                 confs = kp.conf.cpu().numpy()
#                 print(coords[0]) # (17,2); 17 (x,y) keypoints
#                 print(confs[0]) # (17,)

#                 for (x,y), c in zip(coords[0], confs[0]):
#                     if c > 0.4:
#                         cv2.circle(frame, (int(x), int(y)), 3, (0,255,0), -1)

#             cv2.imshow("pose estimation", frame)
#             if cv2.waitKey(1) & 0xFF==ord("q"):
#                 break

#         self.cap.release()
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     pose_estimator = PoseEstimator(
#         model_path="models/yolov8m-pose.pt",
#         video_path="sources/game_clip2.mp4"
#     )
#     pose_estimator.estimate()




# # ------quick test-------
# # Load a model
# model = YOLO("models/yolov8m-pose.pt")  # load an official model

# # Predict with the model
# results = model("sources/game1.jpg")  # predict on an image

# # Access the results
# for result in results:
#     xy = result.keypoints.xy  # x and y coordinates
#     xyn = result.keypoints.xyn  # normalized
#     kpts = result.keypoints.data  # x, y, visibility (if available)
#     result.show()