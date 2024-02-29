import json

from psnr_motion_detector import PSNRMotionDetector
from rt_processor import LiveVideoProcessor
from kcf_multi_tracker import KCFMultiTracker
from mecfTracker import mecfTracker


def main(video_file_path: str, config_file_path: str):
    """ Runs the object detection and tracking """
    # Open the config
    with open(config_file_path, 'r') as fh:
        config = json.load(fh)

    # Initialize the detector and the tracker
    detector = PSNRMotionDetector(
        sample_factor=1,
        bb_min_width=15,
        bb_min_height=15,
        rois_max_diff=4,
        **config["detector"]
    )

    tracker = KCFMultiTracker(
        tracker_class=mecfTracker,
        iou_threshold_detector=0.15,
        iou_threshold_tracker=0.001,
        reconfirmation_threshold=100,
        reconfirmation_counter_with_roi=10,
        reconfirmation_counter_without_roi=1,
        keep_rois_mode="new",    # either old or new
        print_terminations=False,
        **config["tracker"]
    )

    # Initialize the processor
    processor = LiveVideoProcessor(detector=detector, tracker=tracker)

    # Run the live tracking
    processor.live_tracking(
        file_path=video_file_path,
        display_live=True,
        debug=False,
        stream_speed=2,
        **config["processor"]
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("video_file_path", help="path to video file", type=str)
    parser.add_argument("config_file", help="path to config file", type=str)
    args = parser.parse_args()
    video_file_path = args.video_file_path
    config_file_path = args.config_file

    main(video_file_path, config_file_path)
