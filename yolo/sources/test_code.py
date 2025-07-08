from ultralytics import YOLO
import supervision as sv
from trackers import SORTTracker
from trackers import DeepSORTTracker
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = YOLO("models/yolov8x-pose.pt")

tracker = sv.ByteTrack(lost_track_buffer=80,
                                    minimum_matching_threshold=0.6,
                                    track_activation_threshold=0.3,
                                    minimum_consecutive_frames=3)

annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

# tracker = SORTTracker(lost_track_buffer=90,
#                       track_activation_threshold=0.25,
#                       minimum_iou_threshold=0.25) 

# annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

# # results = model.track("sources/2man_game2.mp4", show=True)


# def callback(frame, _):
#     result = model(frame)[0]
#     detections = sv.Detections.from_ultralytics(result)
#     detections = tracker.update(detections)
#     return annotator.annotate(frame, detections, labels=detections.tracker_id)

# sv.process_video(
#     source_path="sources/game_clip4.mp4",
#     target_path="results/test_tracker.mp4",
#     callback=callback,
# )