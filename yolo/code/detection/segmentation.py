from ultralytics import YOLO
import cv2
import torch

model = YOLO("models/yolov8m-seg.pt")

# open the video path
video_path = "sources/background video _ people _ walking _.mp4"

cap = cv2.VideoCapture(video_path) # OpenCV的视频读取器
#cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read() # each time read 1 frame

    if success:
        # run YOLOv8 inference on the frame
        results = model(frame, classes=[0, 32])

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 segment", annotated_frame)
        
        for r in results:
            print(r.boxes)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

    else:
        break

# release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
