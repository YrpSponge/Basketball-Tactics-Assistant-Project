from .player_tracking import PlayerTracking
from .pose_estimation import PoseEstimator
import torch
import cv2
import numpy as np
import os
import json

class InfoCollector:
    def __init__(self, video_path, track_model_path, pose_model_path, output_dir):
        self.track_model = Tracking(track_model_path)
        self.pose_model = PoseEstimator(pose_model_path)
        self.cap = cv2.VideoCapture(video_path)

        # create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


        self.results = {}
        self.frame_idx = 0
    
    
    # --------core function---------
    def collect(self):
        while self.cap.isOpened():
            sucess, frame = self.cap.read()
            if not sucess:
                break
            
            # Tracker
            track_results = self.track_model.track(frame)
            # Pose estimation
            pose_results = self.pose_model.estimate(frame)

            info = []
            # for each tracked person
            for trk in track_results:
                cx, cy = trk["center"]
                # find the closest pose to the center of the person
                closest_pose = min(pose_results, key=lambda p: np.linalg.norm(np.array(p["center"])-np.array([cx,cy])))
                # update the info dictionary of the person: add 2 keys
                trk.update({
                    "pose": closest_pose["keypoints"],
                    "pose_confs": closest_pose["confs"]
                }) 
                info.append(trk)

            self.results[f"frame_{self.frame_idx:04d}"] = info # collect by frame_idx
            self.frame_idx += 1

            self.visualize(frame, info)
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.save_data_by_ID()
        # self.save_data_by_frame()


    def visualize(self, frame, info): # visualize the keypoints on the target
        for obj in info:
            # draw keypoints
            for(x, y), conf in zip(obj["pose"], obj["pose_confs"]):
                if conf > 0.4:
                    cv2.circle(frame, center=(int(x), int(y)), radius=3, color=(0,255,0), thickness=-1)

            # draw bbox
            x1,y1,x2,y2 = obj["bbox"]
            cv2.rectangle(frame, pt1=(int(x1),int(y1)), pt2=(int(x2),int(y2)), color=(255, 0, 0), thickness=2)

            # draw ID
            track_id = obj["id"]
            cv2.putText(
                frame, 
                text=f"ID: {track_id}", 
                org=(int(x1), int(y1-10)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale=0.5, 
                color=(0,0,255),
                thickness=2
                )

        cv2.imshow("Track + Pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


    def save_data_by_ID(self): # save the sata as JSON file
        # collect the data by ID
        id_based_info = {}

        for frame_key, objs in self.results.items():
            frame_num = int(frame_key.split("_")[1])
            for obj in objs:
                track_id = obj["id"]
                entry = {
                    "conf": obj["conf"],
                    "frame": frame_num,
                    "bbox": obj["bbox"],
                    "center": obj["center"],
                    "class": obj["class"],
                    "pose": obj["pose"],
                    "pose_confs": obj["pose_confs"]
                }            

                if track_id not in id_based_info:
                    id_based_info[track_id] = []

                id_based_info[track_id].append(entry)
        
        output_path = os.path.join(self.output_dir, "info_collection_by_ID.json")
        with open(output_path, "w") as f:
            json.dump(id_based_info, f, indent=2)     
            
            print(f"✅ Tracking results saved to {self.output_dir}/info_collection_by_ID.json")

    def save_data_by_frame(self): # save the sata as JSON file
        with open(os.path.join(self.output_dir, "info_collection_by_frame.json"), "w") as f:
            json.dump(self.results, f, indent=2)

            print(f"✅ Tracking results saved to {self.output_dir}/info_collection_by_frame.json")