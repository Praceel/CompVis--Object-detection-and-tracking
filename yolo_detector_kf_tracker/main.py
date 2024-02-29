from rt_processor import LiveVideoProcessor
import os
import sys
root = os.path.dirname(sys.path[0])
sys.path.append(root)
from sort import Sort
from detect import YOLO


def main(video_file_path: str, config_file_path: str):
""" Runs the object detection and tracking """

    detector = YOLO("yolo_param.pt") #load yolo detector model 

    tracker = Sort(
        max_age=250, 
        min_hits=3,
        iou_threshold=0.01
    )
    processor = LiveVideoProcessor(
        detector=detector,
        tracker=tracker
    )

    processor.live_tracking(
    file_path= video_file_path,   # Insert video path here
    save_video=True,
    display_live=False,
    debug=False
    )
    
    
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_file_path", help="path to video file", type=str)
    args = parser.parse_args()
    video_file_path = args.video_file_path
    main(video_file_path)
