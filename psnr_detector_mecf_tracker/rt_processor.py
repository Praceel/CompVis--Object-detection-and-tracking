"""
This is an implementation of a general class for real-time object detection and tracking.
Custom tracker and object detector should be passed to this class at its initialization.

Authors: Hamideh Ayati, Vojtech Vlcek, Jenish Thapa, Prasil Adhikari, Philipp Zwetkoff
"""
import cv2 as cv
import numpy as np


# --------------- Processor Class ---------------
class LiveVideoProcessor:
    """ A class for live object detection and tracking in a video """

    def __init__(self, detector: callable, tracker: callable = None):
        """ Init of the video class """
        # Save the instance of detector and tracker
        self.detector = detector
        self.tracker = tracker

    def live_tracking(self, file_path: str, save_video: bool = True, display_live: bool = True,
                      stream_speed: int = 1, debug: bool = False):
        """ Method for live object tracking from a video """
        # Start the video instance
        video = cv.VideoCapture(file_path)

        # Prepare empty list for saving frames, if requested
        if save_video:
            output_frames = []

        # Loop over the frames
        i = 1
        while True:
            ret, frame = video.read()

            # If a frame is read correctly, ret is True
            if not ret:
                break

            # Do for the first frame
            if i == 1:
                # Blank and dictionary to draw the tracks
                blank = np.zeros_like(frame, dtype=np.uint8)
                dict_tracks = {}
                tracks = None

                # Initial detection
                if self.detector.group_size != 1:
                    rois = self.detector(frame)

            # Synchronization of the tracker and object detector (based on detectors frame groupings)
            # ==> tracker will be called only when the detector's predictions changes
            if i % self.detector.group_size == 0:
                # Object detection
                rois = self.detector(frame)

                # Start tracking, if tracker provided
                if self.tracker is not None:
                    tracks = self.tracker(frame, rois)
            # Else ignore the prediction (keep the same as list time
            else:
                _ = self.detector(frame)

            # Drawing the tracks
            if tracks is not None:
                for track in tracks:
                    ids = track[4]
                    # check it's not placeholder id of unmatched rois
                    if ids != 0:
                        x, y, w, h = track[:4]
                        center = (int(x + w/2), int(y + h/2))
                        if ids in dict_tracks.keys():
                            cv.line(blank, dict_tracks[ids], center, (255, 255, 255), 1)
                        elif debug: 
                            track_id = str(ids)
                            cv.putText(blank, track_id, center, cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

                        dict_tracks[ids] = center
                
                frame = cv.add(blank, frame)
                frame = self.draw_tracks(frame, tracks, rgb=(0, 255, 0), debug=debug)
            else:
                frame = self.draw_bounding_boxes(frame, rois)
            
            # If the video should be shown live
            if display_live:
                cv.imshow("Output", frame)

                key = cv.waitKey(stream_speed) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # Save the frame, if requested
            if save_video:
                output_frames.append(frame)

            # Next frame
            i += 1

        # Save the video, if specified mp4
        if save_video:
            out_path = file_path.split(".")[0] + "_processed.avi"
            self.save_video(np.array(output_frames, dtype="uint8"), out_path)

        # Exit the open video
        video.release()

    @staticmethod
    def save_video(video: np.ndarray, file_name: str, frames_per_sec: float = 30.0, decrease: int = 0):
        """ Saves the processed video """
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter(
            file_name,
            fourcc,
            frames_per_sec,
            (video.shape[-2], video.shape[-3])
        )

        # Remove some frames for smaller video size
        for i, frame in enumerate(video):
            # if i % decrease == 0:
            out.write(frame)

    @staticmethod
    def draw_bounding_boxes(frame: np.ndarray, rois: list, rgb: tuple = (0, 0, 255)):
        """ Draws bounding boxes """
        for bb in rois:
            x, y, w, h = bb
            cv.rectangle(frame, (x, y), (x + w, y + h), rgb, 1)

        return frame
        
    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: list, rgb: tuple=(0,255,0), debug: bool = False):
        """ Draws bounding boxes of tracked detections """
        for bb in tracks:
            x, y, w, h = bb[:4]
            if debug:
                track_id = str(bb[4])
                cv.putText(frame, track_id, (x, y-5), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv.rectangle(frame, (x, y), (x + w, y + h), rgb, 1)

        return frame
