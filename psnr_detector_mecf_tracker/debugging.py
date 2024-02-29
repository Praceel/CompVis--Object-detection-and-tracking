"""
The script serves for debugging purposes.
"""
from psnr_motion_detector import PSNRMotionDetector
from rt_processor import LiveVideoProcessor
from kcf_multi_tracker import KCFMultiTracker
from mecfTracker import mecfTracker

# Press 'q' to stop the live stream
# Good parameters 8, 2, 25, 0.3, 1, 15, 15, 4
detector = PSNRMotionDetector(
    window_size=8,
    stride=2,
    threshold_psnr=25,
    threshold_act=0.3,
    group_size=5,
    sample_factor=1,
    bb_min_width=15,
    bb_min_height=15,
    rois_max_diff=4
)

# Good parameters mecfTracker, True, True, 0.5, 0.22, 0.15, 0.001, 150, 5, 1, new
# but problem with bboxes "running away"
tracker = KCFMultiTracker(
    tracker_class=mecfTracker,
    fixed_window=True,
    multiscale=True,
    occlusion_threshold=0.50,   # Lower when the bounding boxed "run away"
    stopping_threshold=0.22,    # Tweak when you want to terminate bounding boxed with low "confidence"
    iou_threshold_detector=0.15,
    iou_threshold_tracker=0.001,
    reconfirmation_threshold=100,
    reconfirmation_counter_with_roi=10,
    reconfirmation_counter_without_roi=1,
    keep_rois_mode="new",    # either old or new
    print_terminations=True
)

processor = LiveVideoProcessor(
    detector=detector,
    tracker=tracker
)
processor.live_tracking(
    file_path="/home/vojtech/Documents/skola_dokumenty/johannes_kepler_university"
              "/zimni_semestr_2021-2023/Computer Vision/project/data_target_tracking/video_014.mp4",
    save_video=False,
    display_live=True,
    stream_speed=1,  # The lower, the faster (int)
    debug=False
)

# When the method completely fails: 21, 24
