import os

# 获取 config.py 所在目录的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"current dir: {current_dir}")

# 项目根目录（code 的上级目录，即 YOLO 根目录）
project_root = os.path.dirname(os.path.dirname(current_dir))
print(f"project root: {project_root}")

# 视频名
video_name = 'game_clip5.mp4'
video_base_name = os.path.splitext(video_name)[0]

# 拼接各文件/目录的绝对路径------------------------------------------

# 输入视频路径（以 game_clip1.mp4 为例，根据实际视频名调整）
INPUT_VIDEO_DIR = os.path.join(project_root, 'sources', video_name)

# 战术板图片路径
TACTIC_BOARD_DIR = os.path.join(project_root, 'sources', 'basketball_court.png')

# 输出结果目录
OUTPUT_VIDEO_DIR = os.path.join(project_root, 'results', f"{video_base_name}_output.mp4")
OUTPUT_INFO_DIR = os.path.join(project_root, 'results', f"{video_base_name}_info.json")

# STUB路径---------------------------------------------
STUBS_DIR = os.path.join(project_root, "stubs", video_base_name) 

# 模型文件路径
#PLAYER_DETECTOR_PATH = os.path.join(project_root, 'models', 'v8l_detection_training(100epochs).pt') # 原模型
PLAYER_DETECTOR_PATH = os.path.join(project_root, 'models', 'v8x_player_ball.pt') # 新训练模型
#BALL_DETECTOR_PATH = os.path.join(project_root, 'models', 'v8l_detection_training(100epochs).pt')  
BALL_DETECTOR_PATH = PLAYER_DETECTOR_PATH
COURT_KEYPOINT_DETECTOR_PATH = os.path.join(project_root, 'models', 'v8x_court_detection.pt')

