import cv2
import torch
from infer_localization import infer_frame

def video_writer(input_video_path, output_video_path, model):
    """Function to write video"""
    video = cv2.VideoCapture(input_video_path)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)

    out_video = cv2.VideoWriter(output_video_path, 
                        cv2.VideoWriter_fourcc(*'MJPG'), 2, size)

    while video.isOpened():
        ret, frame = video.read()
        if ret == True:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = infer_frame(frame, model)
            out_video.write(frame)
        else:
            break

    video.release()
    out_video.release()

model = torch.load('vgg_with_bn1_reg6.pth')
video_writer('../output.mp4', '../video_with_result.mp4', model)