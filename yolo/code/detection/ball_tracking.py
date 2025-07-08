from ultralytics import YOLO
import cv2
import torch
import os
import json
import numpy as np
import supervision as sv 
from utils import read_stub, save_stub


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class BallTracking:
    def __init__(self, model_path:str, batch_size = 20):
        self.model = YOLO(model_path)
        self.model.to('cuda')
        self.tracker = sv.ByteTrack(lost_track_buffer=60,
                                    minimum_matching_threshold=0.4,
                                    track_activation_threshold=0.25
                                    ) # 调整参数来增强跟踪的稳定性
        self.batch_size = batch_size
    

    def detect_frames(self, frames):
        '''
        Ags:
            frames(list): List of video frmaes to process

        Returns:
            list: YOLO detection results for each frame
        '''
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(frames[i:i+self.batch_size], conf=0.1) # spit into small batch
            detections += detections_batch
        return detections


    def track_ball(self, frames, read_from_stub=False, stub_path=None):
        #--------try to read cache---------
        tracked = read_stub(read_from_stub, stub_path)
        if tracked is not None:
            if len(tracked) == len(frames): # if info in all the frames is handled
                return tracked
        
        detections = self.detect_frames(frames)

        tracked = []

        for frame_id, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()} # 希望name作为key, 编号作为value

            # convert to supervision Detection format
            detection_sv = sv.Detections.from_ultralytics(detection)

            # # track ball-----choose the detected object with max confidence 方法一；一帧帧找最高置信度的ball，前后帧的关联差，效果一般
            
            # tracked.append({})
            # max_confidence = 0
            # chosen_ball = None

            # for frame_detection in detection_sv:
            #     bbox = frame_detection[0].tolist()
            #     cls_id = frame_detection[3]
            #     confidence = frame_detection[2]
                
                
            #     # Track the ball with max confidence
            #     if cls_id == cls_names_inv['Ball']:
            #         if confidence > max_confidence: # update the max-conf object
            #             chosen_ball = bbox
            #             max_confidence = confidence

            # if chosen_ball is not None: 
            #     x1,y1,x2,y2 = chosen_ball
            #     cx,cy = (x1+x2)/2, (y1+y2)/2 
            #     center = [round(cx, 2), round(cy, 2)] # center of the bbox
            #     tracked[frame_id][1] = {"bbox": chosen_ball, "center": center}
            # #-------------------------------------------------------------------------------------------------------------------------

            # track ball-----ID关联 方法二；
            tracked.append({})
            tracked_detections = self.tracker.update_with_detections(detection_sv)
            curr_frame_results = {}
            ball = None

            for obj_data in tracked_detections:
                bbox = obj_data[0].tolist()
                cls_id = obj_data[3]
                track_id = obj_data[4]

                if cls_id==cls_names_inv['Ball']:
                    ball = bbox

            if ball is not None:
                x1, y1, x2, y2 = ball
                cx,cy = (x1+x2)/2, (y1+y2)/2 
                center = [round(cx, 2), round(cy, 2)] # center of the bbox
                tracked[frame_id][track_id] = {"bbox": ball, "center": center, "cls": 'Ball'}
            # --------------------------------------------------------------------------------------------------------------------------

        
        save_stub(stub_path, tracked)
        return tracked