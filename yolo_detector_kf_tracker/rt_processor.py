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

    def live_tracking(self, file_path: str, save_video: bool = True, display_live: bool = True, debug: bool = False):
        """ Method for live object tracking from a video """
        # Start the video instance
    
        video = cv.VideoCapture(file_path)

        # Prepare empty list for saving frames, if requested
        if save_video:
            output_frames = []
            
        # Blank and dictionary to draw the tracks
        blank = np.zeros((512,640,3), dtype=np.uint8)
        dict = {}

        # Loop over the frames
        while True:
            ret, frame = video.read()
         

            # If a frame is read correctly, ret is True
            if not ret:
                break

            # Start object detection
            rois = self.detector(frame)
       

            # Start tracking, if tracker provided
            if self.tracker is not None:
                detections = np.array(rois) # detections of form [x1,y1,x2,y2,]

                matched_dets, unmatched_preds, unmatched_dets = self.tracker.update(detections.astype(np.int32))
                if len(unmatched_preds) > 0:
                    tracks = np.concatenate((matched_dets, unmatched_preds), axis=0)
                else:
                    tracks = matched_dets
                for track in tracks:
                    id = track[4]
                    # check it's not placeholder id of unmatched rois
                    if id != 0:
                        x1, y1, x2, y2 = track[:4]
                        center = (int((x1 + x2)/2), int((y1 + y2)/2))
                        if id in dict.keys():
                            cv.line(blank, dict[id], center, (255,255,255), 1)
                        elif debug: 
                            track_id = str(int(id))
                            cv.putText(blank, track_id, center, cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                        dict[id] = center
                frame = cv.add(blank, frame)
                # Draw the bounding boxes
                if debug:
                    frame = self.draw_bounding_boxes(frame, unmatched_dets, color=(255,0,0))
                    frame = self.draw_tracks(frame, unmatched_preds, color=(0,0,255), debug=debug)
                    frame = self.draw_tracks(frame, matched_dets, color=(0,255,0), debug=debug)
                else:
                    frame = self.draw_tracks(frame, tracks)
            else:
                frame = self.draw_bounding_boxes(frame, rois)

            # If the video should be shown live
            if display_live:
                cv.imshow("Output", frame)

                key = cv.waitKey(5) & 0xFF
                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break

            # Save the frame, if requested
            if save_video:
                output_frames.append(frame)

        # Save the video, if specified mp4
        if save_video:
            out_path = file_path.split(".mp4")[0] + "_processed.avi"
            print("Video saved at ", out_path)
            self.save_video(np.array(output_frames, dtype="uint8"), out_path)

        # Exit the open video
        video.release()

    @staticmethod
    def save_video(video: np.ndarray, file_name: str, frames_per_sec: float = 28.0):
        """ Saves the processed video """
        
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        out = cv.VideoWriter(
            file_name,
            fourcc,
            frames_per_sec,
            (video.shape[-2], video.shape[-3])
        )

        for frame in video:
            out.write(frame)
        out.release()

    @staticmethod
    def draw_bounding_boxes(frame: np.ndarray, rois: list, color: tuple=(0,0,255)):
        """ Draws bounding boxes """
        for bb in rois:
            x1, y1, x2, y2 = bb.astype(np.int32)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        return frame
        
    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: np.ndarray, color: tuple=(0,255,0), debug: bool = False):
        """ Draws bounding boxes of tracked detections """
        for bb in tracks:
            bb = bb.astype(np.int32)
            x1, y1, x2, y2 = bb[:4]
            if debug:
                track_id = str(bb[4])
                cv.putText(frame, track_id, (x1, y1-5), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        return frame