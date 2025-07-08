from ultralytics import YOLO
import cv2
import torch
import os
import json
import numpy as np
import supervision as sv 

from torchreid.reid.utils import FeatureExtractor
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from utils import read_stub, save_stub

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class PlayerTracking:
    '''
    A class that detect and track players
    '''
    def __init__(self, model_path:str, batch_size = 20):
        self.model = YOLO(model_path)
        self.model.to('cuda')
        self.tracker = sv.ByteTrack(lost_track_buffer=80,
                                    minimum_matching_threshold=0.6,
                                    track_activation_threshold=0.3,
                                    minimum_consecutive_frames=3)
        self.batch_size = batch_size

        self.reid_model = FeatureExtractor(
            model_name='osnet_x1_0', 
            device='cuda',
            model_path=None  # 若使用torchreid预训练模型，留空即可
            )

    def detect_frames(self, frames):
        '''
        Ags:
            frames(list): List of video frmaes to process

        Returns:
            list: YOLO detection results for each frame
        '''
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(frames[i:i+self.batch_size], conf=0.1) # spit into small batch，降低conf让ByteTrack去二级关联
            detections += detections_batch

        return detections
            



    def track_player(self, frames, read_from_stub=False, stub_path=None):
        '''
        Get tracking results for a sequence of frames with optional caching

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box + center coordinates.
                [
                {id1:{"bbox": [x,y,x,y], "center": [x,y]}, id2:{"bbox": [x,y,x,y], "center": [x,y]}}, # frame 1
                {id1:                                    , id2:                   }, # frame 2
                ...
                {.................................................................} # frame k
                ]
        '''
        
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

            # track player
            detection_with_tracks = self.tracker.update_with_detections(detection_sv) # get a tuple list
            #print(detection_with_tracks) #test
            
            tracked.append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                x1,y1,x2,y2 = bbox
                cx,cy = (x1+x2)/2, (y1+y2)/2 
                center = [round(cx, 2), round(cy, 2)] # center of the bbox

                if cls_id == cls_names_inv['Player']:
                    cropped = self.crop_person(frames[frame_id], bbox)
                    feature = self.reid_model(cropped)[0]
                    feature = feature.tolist()


                    tracked[frame_id][track_id] = {"bbox": bbox, 
                                                   "center": center,
                                                   "feature": feature,
                                                   "cls": 'Player'}
        

        tracked = self.merge_similar_ids_by_reid(tracked)
        save_stub(stub_path, tracked)

        return tracked
        
        
    def crop_person(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)
        cropped = frame[y1:y2, x1:x2]
        return cropped
    
    
    def merge_similar_ids_by_reid(self, tracked, sim_threshold=0.9):
        """
        tracked: List[Dict[int, Dict[str, Any]]] — 帧列表，每帧为 ID 到对象信息的字典，需含有 "reid" 键
        sim_threshold: float — 余弦相似度阈值，高于此值将认为两个 ID 是同一个人
        """
        # Step 1: 聚合每个 ID 的 ReID 向量列表
        id_to_features = defaultdict(list)
        for frame in tracked:
            for id_, info in frame.items():
                if "reid" in info:
                    id_to_features[id_].append(info["reid"])

        # Step 2: 计算每个 ID 的平均特征向量
        id_to_avg = {
            id_: np.mean(np.stack(feats), axis=0)
            for id_, feats in id_to_features.items() if len(feats) > 0
        }

        # Step 3: 构建合并映射（合并重复 ID）
        id_list = list(id_to_avg.keys())
        merged_map = {}

        for i in range(len(id_list)):
            for j in range(i + 1, len(id_list)):
                id1, id2 = id_list[i], id_list[j]
                feat1, feat2 = id_to_avg[id1].reshape(1, -1), id_to_avg[id2].reshape(1, -1)
                sim = cosine_similarity(feat1, feat2)[0][0]
                if sim >= sim_threshold:
                    # 优先保留较小的 ID（也可以按 track 长度排序后保留长轨迹的）
                    keep_id = min(id1, id2)
                    merge_id = max(id1, id2)
                    merged_map[merge_id] = keep_id

        # Step 4: 替换合并后的 ID
        def resolve_final_id(id_):
            # 避免链式合并：a→b→c
            while id_ in merged_map:
                id_ = merged_map[id_]
            return id_

        for frame in tracked:
            new_frame = {}
            for id_, info in frame.items():
                new_id = resolve_final_id(id_)
                if new_id not in new_frame:
                    new_frame[new_id] = info
                else:
                    # 可以选择合并两个 ID 的特征或保留一个
                    pass
            frame.clear()
            frame.update(new_frame)

        return tracked

        
        
        







        # # go over all the boxes in a frame
        # for box in results[0].boxes:
        #     if box.id is None:
        #         continue
        #     x1,y1,x2,y2 = box.xyxy[0].tolist()
        #     cx,cy = (x1+x2)/2, (y1+y2)/2 # center of the bbox

        #     # use a dictionary to store the info of one box
        #     dict_for_a_box = {
        #         "id": int(box.id.item()),
        #         "conf": round(float(box.conf.item()), 3),
        #         "bbox":[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        #         "center": [round(cx, 1), round(cy, 1)],
        #         "class": int(box.cls.item())
        #     }

        #     tracked.append(dict_for_a_box)

        # return tracked







































    # -------old version------------
    # def track(self):
    #     while self.cap.isOpened():
    #         success, frame = self.cap.read()
    #         if not success:
    #             break

    #         results = self.model.track(frame, persist=True, classes=[0,32], conf=0.3, tracker="bytetrack.yaml")

    #         self.display(results)

    #         # collect data of each box in the frame
    #         frame_data = []
    #         for box in results[0].boxes:
    #             if box.id is None:
    #                 continue # skip the object without id
                
    #             # key data for each box
    #             track_id = int(box.id.item())
    #             conf = round(float(box.conf.item()), 3) # round(原数， 保留的位数)
    #             x1, y1, x2, y2 = box.xyxy[0].tolist() # box's corners
    #             cx, cy = round((x1+x2)/2, 1), round((y1+y2)/2, 1) # box‘s center
    #             obj_cls = int(box.cls.item())

    #             dict_for_a_box={
    #                 "id":track_id,
    #                 "conf":conf,
    #                 "bbox":[round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
    #                 "center": [cx,cy],
    #                 "class": obj_cls
    #             }

    #             frame_data.append(dict_for_a_box)
    #             print(dict_for_a_box)

    #         self.tracks_dict[f"frame_{self.frame_idx: 04d}"] = frame_data
    #         self.frame_idx += 1
        
    #     self.cap.release()
    #     self.save_tracking_data()


    # def save_tracking_data(self):
    #     # save tracking data as JSON file
    #     with open(os.path.join(self.output_dir, "tracking_results.json"), "w") as f:
    #         json.dump(self.tracks_dict, f, indent=2)

    #     print(f"✅ Tracking results saved to {self.output_dir}/tracking_results.json")


    # def display(self, results):
    #     # Visualize the results on the frame
    #     annotated_frame = results[0].plot()

    #     # Display the annotated frame
    #     cv2.imshow("YOLOv8 segment", annotated_frame)
        
    #     # for r in results:
    #     #     print(r.boxes)

    #     # break the loop if 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF==ord("q"):
    #         return

    # def visualize(self):
    #     # 加载追踪数据
    #     with open(os.path.join(self.output_dir, "tracking_results.json"), "r") as f:
    #         track_data = json.load(f)

    #     # 存储每个 ID 的轨迹点
    #     track_points = {}

    #     for frame_key in sorted(track_data.keys()):
    #         for obj in track_data[frame_key]:
    #             track_id = obj["id"]
    #             class_id = obj["class"]
    #             center = obj["center"]
    #             if track_id not in track_points:
    #                 track_points[track_id] = {"points": [], "class": class_id}
    #             track_points[track_id]["points"].append(center)

    #     # 创建画布
    #     canvas = 255 * np.ones((720, 1280, 3), dtype=np.uint8)

    #     # 绘制轨迹
    #     for track_id, info in track_points.items():
    #         class_id = info["class"]
    #         points = info["points"]
    #         color = self.get_color_by_class(class_id)
            
    #         for i in range(1, len(points)):
    #             pt1 = tuple(map(int, points[i-1]))
    #             pt2 = tuple(map(int, points[i]))
    #             cv2.line(canvas, pt1, pt2, color, 2)

    #     cv2.imshow("Tracking Trajectories", canvas)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # def get_color_by_class(self, class_id):
    #     if class_id == 0:
    #         return (0, 225, 0)      # person → green
    #     elif class_id == 32:
    #         return (0, 0, 225)      # sports ball → red
    #     else:
    #         return (128, 128, 128)  # 其他 → 灰色

        
# # apply the class
# if __name__=="__main__":
#     print(torch.__version__)          # PyTorch版本
#     print(torch.version.cuda)         # CUDA版本
#     print(cv2.__version__)            # OpenCV版本

#     # init
#     tracker = Tracking(model_path="models/yolov8l.pt", video_path="sources/game_clip3.mp4")

#     # start tracking
#     tracker.track()

#     # visualization
#     tracker.visualize()



