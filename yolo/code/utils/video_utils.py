'''
A module for video file reading, playing and writing

'''

import cv2
import os
import numpy as np

def read_video(video_path):
    '''
    Returns:
        list: List of video frames as np arrays.
    '''
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        sucess, frame = cap.read()
        if not sucess:
            break
        frames.append(frame)

    return frames


def save_video(output_frames, output_path, video_path):
    '''
    Save annotated video
    '''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))



    # if folder does not exist, create it
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    ext = os.path.splitext(video_path)[1].lower()
    if ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'XVID') # for .avi
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # for mp4
    
    out = cv2.VideoWriter(output_path, 
                          fourcc, # 视频编码器
                          fps, # 视频帧率
                          (output_frames[0].shape[1], output_frames[0].shape[0]) # 视频尺寸
                          )    
    if not out.isOpened():
        print("错误：VideoWriter 初始化失败，无法保存视频")
        return False
    

    # 写入帧并检查是否成功
    frame_count = 0
    for i, frame in enumerate(output_frames):
        # 验证帧格式
        if frame.dtype != np.uint8:
            print(f"警告: 第 {i} 帧类型不是uint8，尝试转换")
            frame = (frame * 255).astype(np.uint8)
        
        # 验证帧通道数
        if len(frame.shape) == 2:
            print(f"警告: 第 {i} 帧是灰度图，转换为BGR")
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        out.write(frame)
    
    # 释放资源
    out.release()
    print(f"视频保存成功: {output_path}")
    print(f"总帧数: {len(output_frames)}")
    return True

