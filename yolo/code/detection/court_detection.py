from ultralytics import YOLO
import supervision as sv
import sys 
from utils import read_stub, save_stub 

class CourtDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)


    def get_court_keypoints(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect court keypoints for a batch of frames using the YOLO model. If requested, 
        attempts to read previously detected keypoints from a stub file before running the model.

        Args:
            frames (list of numpy.ndarray): A list of frames (images) on which to detect keypoints.
            read_from_stub (bool, optional): Indicates whether to read keypoints from a stub file 
                instead of running the detection model. Defaults to False.
            stub_path (str, optional): The file path for the stub file. If None, a default path may be used. 
                Defaults to None.

        Returns:
            list: A list of detected keypoints for each input frame.
        """
        court_keypoints = read_stub(read_from_stub,stub_path)
        if court_keypoints is not None:
            if len(court_keypoints) == len(frames):
                return court_keypoints
        
        batch_size=20
        court_keypoints = []
        for i in range(0,len(frames),batch_size): # go over all frames
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.5) # 一批一批detect
            print(detections_batch)
            for detection in detections_batch:
                court_keypoints.append(detection.keypoints)

        save_stub(stub_path,court_keypoints)
        
        return court_keypoints       