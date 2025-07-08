import os
import sys
import pathlib
import numpy as np
import cv2
from copy import deepcopy
from .homography import Homography
from utils.bbox_utils import get_foot_position 


class TacticalViewConverter:
    def __init__(self, court_image_path):
        self.court_image_path = court_image_path
        # pixel size of the court in image
        self.width_in_pixel = 300
        self.height_in_pixel = 161

        # actual size of a court
        self.actual_width_in_meters = 28
        self.actual_height_in_meters = 15
        
        # define the keypoints in the tactical view
        self.keypoints = [
            # left edge
            (0,0),
            (0,int((0.91/self.actual_height_in_meters)*self.height_in_pixel)), # calculate the point in pixel coordinate with proportion 比例尺映射
            (0,int((5.18/self.actual_height_in_meters)*self.height_in_pixel)),
            (0,int((10/self.actual_height_in_meters)*self.height_in_pixel)),
            (0,int((14.1/self.actual_height_in_meters)*self.height_in_pixel)),
            (0,int(self.height_in_pixel)),

            # Middle line
            (int(self.width_in_pixel/2),self.height_in_pixel),
            (int(self.width_in_pixel/2),0),
            
            # Left Free throw line
            (int((5.79/self.actual_width_in_meters)*self.width_in_pixel),int((5.18/self.actual_height_in_meters)*self.height_in_pixel)),
            (int((5.79/self.actual_width_in_meters)*self.width_in_pixel),int((10/self.actual_height_in_meters)*self.height_in_pixel)),

            # right edge
            (self.width_in_pixel,int(self.height_in_pixel)),
            (self.width_in_pixel,int((14.1/self.actual_height_in_meters)*self.height_in_pixel)),
            (self.width_in_pixel,int((10/self.actual_height_in_meters)*self.height_in_pixel)),
            (self.width_in_pixel,int((5.18/self.actual_height_in_meters)*self.height_in_pixel)),
            (self.width_in_pixel,int((0.91/self.actual_height_in_meters)*self.height_in_pixel)),
            (self.width_in_pixel,0),

            # Right Free throw line
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width_in_pixel),int((5.18/self.actual_height_in_meters)*self.height_in_pixel)),
            (int(((self.actual_width_in_meters-5.79)/self.actual_width_in_meters)*self.width_in_pixel),int((10/self.actual_height_in_meters)*self.height_in_pixel)),
        ]
        
    def validate_keypoints(self, keypoints_list):
        '''
        validate detected keypoints by comparing their proportional distance, cancel error match

        Args:
            keypoints_list (List[List[Tuple[float, float]]]): A list containing keypoints for each frame.
                Each outer list represents a frame.
                Each inner list contains keypoints as (x, y) tuples.
                A keypoint of (0, 0) indicates that the keypoint is not detected for that frame.
        
        Returns:
            List[bool]: A list indicating whether each frame's keypoints are valid.
        '''
        
        keypoints_list = deepcopy(keypoints_list)
        print(type(keypoints_list)) # ==List
        #print(keypoints_list) # 发现已经不是Keypoints类型的数据了

        # get thd coord from each frame
        for frame_id, frame_keypoints in enumerate(keypoints_list): # enumerate: 带索引的迭代器
            print(type(frame_keypoints)) # ==dict 好奇怪！！！！
            # print(frame_id)
            # print(frame_keypoints[frame_id+1]["bbox"])
            frame_keypoints = frame_keypoints.xy[0].tolist() # 现在这个frame_keypoints就是 list of keypoints[[x,y], [x.y], [x,y]......]
            '''
            但是，他竟然显示frame_keypoints本身是dict？？？
            '''

            # get keypoints indices (except 0,0)
            detected_indices = [i for i, kp in enumerate(frame_keypoints) if kp[0]>0 and kp[1]>0]

            # Need at least 3 detected keypoints to validate proportions
            if len(detected_indices) < 3:
                continue

            invalid_keypoints = []
            # validate each detected keypoint
            for i in detected_indices:
                # skip (0,0)
                if frame_keypoints[i][0]==0 and frame_keypoints[i][1]==0:
                    continue

                # Choose two other random detected keypoints
                other_indices = [idx for idx in detected_indices if idx != i and idx not in invalid_keypoints]
                if len(other_indices) < 2:
                    continue

                # Take first two other indices for simplicity
                j, k = other_indices[0], other_indices[1]

                # Calculate distances between detected keypoints
                d_ij = np.linalg.norm(np.array(frame_keypoints[i])-np.array(frame_keypoints[j]))
                d_ik = np.linalg.norm(np.array(frame_keypoints[i])-np.array(frame_keypoints[k]))
                
                # Calculate distances between corresponding tactical keypoints
                t_ij = np.linalg.norm(np.array(self.keypoints[i])-np.array(self.keypoints[j]))
                t_ik = np.linalg.norm(np.array(self.keypoints[i])-np.array(self.keypoints[k]))

                # calculate and compare proportions with 50% error margin
                if t_ij > 0 and t_ik > 0:
                    prop_detected = d_ij / d_ik if d_ik > 0 else float('inf')
                    prop_tactical = t_ij / t_ik if t_ik > 0 else float('inf')

                    error = (prop_detected-prop_tactical)/prop_tactical
                    error = abs(error)

                    if error>0.5:
                        keypoints_list[frame_id].xy[0][i] *= 0 # .xy is a tensor of [1,N,2], batch_size=1, kps_num=N, size of (x,y)=2
                        keypoints_list[frame_id].xyn[0][i] *= 0
                        invalid_keypoints.append(i)

        return keypoints_list
    
    def transform_players_to_tactical_view(self, keypoints_list, player_tracks, ball_tracks):
        """
        Transform player positions from video frame coordinates to tactical view coordinates.
        
        Args:
            keypoints_list (list): List of detected court keypoints for each frame.
            player_tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        
        Returns:
            list: List of dictionaries where each dictionary maps player IDs to their (x, y) positions
                in the tactical view coordinate system. The list index corresponds to the frame number.
        """
        tactical_player_positions = []
        
        for frame_idx, (frame_keypoints, player_info, ball_info) in enumerate(zip(keypoints_list, player_tracks, ball_tracks)): # use zip to go over keypoints and player_tracks at the same time
            # Initialize empty dictionary for this frame
            tactical_positions = {}

            # frame
            print(frame_keypoints) # test
            frame_keypoints = frame_keypoints.xy.tolist()[0]

            # Skip frames with insufficient keypoints
            if frame_keypoints is None or len(frame_keypoints) == 0:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Get detected keypoints for this frame
            detected_keypoints = frame_keypoints
            
            # Filter out undetected keypoints (those with coordinates (0,0))
            valid_indices = [i for i, kp in enumerate(detected_keypoints) if kp[0] > 0 and kp[1] > 0]
            
            # Need at least 4 points for a reliable homography
            if len(valid_indices) < 4:
                tactical_player_positions.append(tactical_positions)
                continue
            
            # Create source and target point arrays for homography
            source_points = np.array([detected_keypoints[i] for i in valid_indices], dtype=np.float32) # detected keypoints in the video
            target_points = np.array([self.keypoints[i] for i in valid_indices], dtype=np.float32) # defined keypoints in tactical view
            
            try:
                # Create homography transformer
                homography = Homography(source_points, target_points)
                
                # Transform each player's position
                for player_id, player_data in player_info.items():
                    bbox = player_data["bbox"]
                    # Use bottom center of bounding box as player position
                    player_position = np.array([get_foot_position(bbox)])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(player_position) # what I want!!

                    # If tactical position is not in the tactical view, skip
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width_in_pixel or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height_in_pixel:
                        continue

                    tactical_positions[player_id] = {
                        "center": tactical_position[0].tolist(),
                        "cls": 'Player'
                    }
                
                # Transform ball's position
                for ball_id, ball_data in ball_info.items():
                    bbox = ball_data["bbox"]
                    # Use bottom center of bounding box as player position
                    ball_position = np.array([get_foot_position(bbox)])
                    # Transform to tactical view coordinates
                    tactical_position = homography.transform_points(ball_position) # what I want!!

                    # If tactical position is not in the tactical view, skip
                    if tactical_position[0][0] < 0 or tactical_position[0][0] > self.width_in_pixel or tactical_position[0][1] < 0 or tactical_position[0][1] > self.height_in_pixel:
                        continue

                    tactical_positions[ball_id] = {
                        "center": tactical_position[0].tolist(),
                        "cls": 'Ball'
                    }


            except (ValueError, cv2.error) as e:
                # If homography fails, continue with empty dictionary
                pass
            
            tactical_player_positions.append(tactical_positions)
        
        return tactical_player_positions