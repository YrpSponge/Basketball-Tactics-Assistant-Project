import os
import cv2
import numpy as np
import json


# def save_data_by_ID(self): # save the sata as JSON file
#         # collect the data by ID
#         id_based_info = {}

#         for frame_key, objs in self.results.items():
#             frame_num = int(frame_key.split("_")[1])
#             for obj in objs:
#                 track_id = obj["id"]
#                 entry = {
#                     "conf": obj["conf"],
#                     "frame": frame_num,
#                     "bbox": obj["bbox"],
#                     "center": obj["center"],
#                     "class": obj["class"],
#                     "pose": obj["pose"],
#                     "pose_confs": obj["pose_confs"]
#                 }            

#                 if track_id not in id_based_info:
#                     id_based_info[track_id] = []

#                 id_based_info[track_id].append(entry)
        
#         output_path = os.path.join(self.output_dir, "info_collection_by_ID.json")
#         with open(output_path, "w") as f:
#             json.dump(id_based_info, f, indent=2)     
            
#             print(f"✅ Tracking results saved to {self.output_dir}/info_collection_by_ID.json")

# def save_data_by_frame(self): # save the sata as JSON file
#     with open(os.path.join(self.output_dir, "info_collection_by_frame.json"), "w") as f:
#         json.dump(self.results, f, indent=2)

#         print(f"✅ Tracking results saved to {self.output_dir}/info_collection_by_frame.json")

def save_info(tactical_positions, team_assignment, output_path):
    """
    Save tactical view positions into a structured JSON file.

    Args:
        tactical_positions (list): 每一帧的追踪结果，格式为 [{id: {"cls": ..., "center": [...]}, ...}, ...]
        team_assignment (dict): 对象ID -> 所属阵营，例如 {1: "red", 2: "blue", 99: "neutral"} 或 {"ball": "neutral"}
        output_path (str): 保存的文件路径，例如 "./outputs/track_result.json"
    """

    output_info={}

    for i, frame_data in enumerate(tactical_positions):
        frame_id = f"frame_{i}"
        for obj_id, obj_info in frame_data.items():
            team = team_assignment[i].get(obj_id, 1)
            
            output_info[frame_id] = {
                "obj_id": int(obj_id),
                "cls": obj_info["cls"],
                "team": team,
                "center": obj_info["center"]
            }

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存为 JSON 文件
    with open(output_path, "w") as f:
        json.dump(output_info, f, indent=2)

    print(f"✅ Tactical tracking info saved to {output_path}")
     