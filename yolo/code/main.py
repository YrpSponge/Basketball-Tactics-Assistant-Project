import torch
import cv2
import os
import argparse

from detection import (
    BallTracking,
    CourtDetection,
    InfoCollector,
    PlayerTracking,
    PoseEstimator
)
from perspective_transform import TacticalViewConverter
from configs import(
    PLAYER_DETECTOR_PATH,
    BALL_DETECTOR_PATH,
    COURT_KEYPOINT_DETECTOR_PATH,
    INPUT_VIDEO_DIR,
    OUTPUT_VIDEO_DIR,
    STUBS_DIR,
    TACTIC_BOARD_DIR,
    OUTPUT_INFO_DIR
)
from utils import read_video, save_video, save_info
from visualization import PlayerTracksDrawer, BallTracksDrawer, TacticalViewDrawer
from team_assignment import TeamAssigner

def parse_args():
    parser = argparse.ArgumentParser(description='Basketball Video Analysis')
    parser.add_argument('--input_video', type=str, default=INPUT_VIDEO_DIR, help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=OUTPUT_VIDEO_DIR, help='Path to output annotated video file')
    parser.add_argument('--stub_path', type=str, default=STUBS_DIR, help='Path to stub directory')
    parser.add_argument('--output_info', type=str, default=OUTPUT_INFO_DIR, help='Path to stub directory')
    
    return parser.parse_args()



def main():
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("OpenCV:", cv2.__version__)

    args = parse_args()

    # Read video
    video_frames = read_video(args.input_video)



    # Player tracker
    player_tracker = PlayerTracking(PLAYER_DETECTOR_PATH)
    player_tracks = player_tracker.track_player(video_frames,
                                                read_from_stub=True,
                                                stub_path=os.path.join(args.stub_path, "player_tracks_stub.pkl")
                                                )

    # Ball tracker
    ball_tracker = BallTracking(BALL_DETECTOR_PATH)
    ball_tracks = ball_tracker.track_ball(video_frames,
                                          read_from_stub=True,
                                          stub_path=os.path.join(args.stub_path, "ball_tracks_stub.pkl")
                                          )

    
    # Court detection
    court_keypoints_detector = CourtDetection(COURT_KEYPOINT_DETECTOR_PATH)
    court_keypoints_per_frame = court_keypoints_detector.get_court_keypoints(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path=os.path.join(args.stub_path, "court_keypoints_stub.pkl")
                                                                              )

    # Pose recognition


    # Team assign
    team_assigner = TeamAssigner()

    player_assignment = team_assigner.get_player_teams_across_frames(video_frames,
                                                                     player_tracks,
                                                                     read_from_stub=True,
                                                                     stub_path=os.path.join(args.stub_path, "player_assignment_stub.pkl"))

    # create an collector object: synthesize the track, pose info of a player 考虑要改掉？？
    # info_collector = InfoCollector()
    # info_collector.collect()

    # Tactical View
    tactical_viewer = TacticalViewConverter(TACTIC_BOARD_DIR)

    #---------暂时格式有问题----------数据类型对不上
    print(type(court_keypoints_per_frame))
    print("------------------------------------------------")
    # print(court_keypoints_per_frame) # 发现这里就已经不是Keypoints类型的数据了
    court_keypoints_per_frame = tactical_viewer.validate_keypoints(court_keypoints_per_frame)
 

    tactical_player_positions = tactical_viewer.transform_players_to_tactical_view(court_keypoints_per_frame, player_tracks, ball_tracks)


    # Speed and distance calculation

    # Visualization
    player_tracks_drawer = PlayerTracksDrawer()
    ball_tracks_drawer = BallTracksDrawer()
    tactical_view_drawer = TacticalViewDrawer()

    #output_frames = player_tracks_drawer.draw(video_frames, player_tracks)
    output_frames = player_tracks_drawer.draw(video_frames, player_tracks, player_assignment) # 暂时省略分队的版本
    output_frames = ball_tracks_drawer.draw(output_frames, ball_tracks)
    output_frames = tactical_view_drawer.draw(output_frames, 
                                              tactical_viewer.court_image_path, 
                                              tactical_viewer.width_in_pixel, 
                                              tactical_viewer.height_in_pixel, 
                                              tactical_viewer.keypoints, 
                                              tactical_player_positions,
                                              player_assignment
                                              )


    # save video
    save_video(output_frames, args.output_video, args.input_video)

    # save info of ball & players
    save_info(tactical_player_positions, player_assignment, args.output_info)


if __name__ == "__main__":
    main()


'''
Output:
    # annotated video
    # JSON file: 
        frame_ID: {palyer_ID: [player positions in the tactical view],
                              pose_name
                   ball: [ball positions in the tactical view]
                   }
     
'''