"""
This is an implementation of a tracker class for multi-object tracking.
It is designed to be used with the KCFTracker or mecfTracker.

Authors: Hamideh Ayati, Vojtech Vlcek, Jenish Thapa, Prasil Adhikari, Philipp Zwetkoff
"""
import numpy as np


class KCFMultiTracker:
    def __init__(self, tracker_class: callable, fixed_window: bool, multiscale: bool, occlusion_threshold: float,
                 stopping_threshold: float, iou_threshold_detector: float, iou_threshold_tracker: float,
                 reconfirmation_threshold: int, reconfirmation_counter_with_roi: int,
                 reconfirmation_counter_without_roi: int, keep_rois_mode: str, print_terminations: bool):
        """
        Initialization of the KFCMultiTracker class

        Parameters:
            tracker_class: tracker class KCFTracker or mecfTracker
            fixed_window: if only fixed window should be used for tracking
            multiscale: processing on multiple scales (fixed window is then automatically false)
            occlusion_threshold: if peak value is smaller, motion will be predicted
            stopping_threshold: terminates tracker if its peak values goes below 0.1
            iou_threshold_detector: if overlap between a detector's and a tracker's ROIs is larger ==> no new tracker
            iou_threshold_tracker: if overlap between two trackers' ROIs ==> terminate either the older or newer one
            reconfirmation_threshold: number of 'points' until termination of a tracker
            reconfirmation_counter_with_roi: termination points awarded when detector returns ROIs
            reconfirmation_counter_without_roi: termination points awarded when there are no ROIs from the detector
            keep_rois_mode: if there is overlap between two trackers, which tracker to keep, new or old
            print_terminations: prints when a tracker is terminated and the reason for termination
        """
        # The tracker class that will be initialized
        self.tracker_class = tracker_class

        # Hyperparameters
        self.fixed_window = fixed_window
        self.multiscale = multiscale
        self.occlusion_threshold = occlusion_threshold
        self.stopping_threshold = stopping_threshold
        self.iou_threshold_detector = iou_threshold_detector
        self.iou_threshold_tracker = iou_threshold_tracker
        self.reconfirmation_threshold = reconfirmation_threshold
        self.reconfirmation_counter_with_roi = reconfirmation_counter_with_roi
        self.reconfirmation_counter_without_roi = reconfirmation_counter_without_roi
        self.mode = keep_rois_mode  # Either old or new

        self.print_terminations = print_terminations

        # Buffers - for keeping tracker instances
        self.tracker_instances = []

        # Buffers - past predictions
        self.without_reconfirmation = []

    def __call__(self, frame: np.ndarray, rois: list):
        """ When the tracker is called """
        predictions = []    # empty list for saving predictions

        # If the tracker instances exist --> predict
        if self.tracker_instances:
            # Terminate instance without reconfirmation
            terminate = []
            for i in range(len(self.tracker_instances)):
                if self.without_reconfirmation[i] > self.reconfirmation_threshold:
                    terminate.append(i)
                    if self.print_terminations:
                        print("Deleted - reconfirmation ")

            self._terminate_trackers(terminate)

            # Prepare empty list of instances to be terminated based on peak
            terminate = []

            # Loop over the tracker instances and predict
            for i, tracker in enumerate(self.tracker_instances):
                prediction, peak = tracker.update(frame)

                # If the peak is under the threshold --> terminate
                if peak < self.stopping_threshold:
                    terminate.append(i)
                    if self.print_terminations:
                        print(f"Deleted - stopping threshold, peak value: {peak}")
                # Else save the bounding boxes
                else:
                    predictions.append(list(map(int, map(np.round, prediction))))

            # Remove the tracker instances which do not return reliable predictions
            self._terminate_trackers(terminate)

            # Check for overlapping predictions from the trackers (if more than 2)
            if len(self.tracker_instances) > 1:
                terminate = []

                # Reformat the predictions
                predictions_ref = np.array(predictions)
                predictions_ref[:, 2:4] += predictions_ref[:, 0:2]

                # Get the iou
                ious = np.array([
                    [get_iou(pred, other_pred) for other_pred in predictions_ref]
                    for pred in predictions_ref
                ])

                # Mode: old will be deleted for the sake of new
                if self.mode == "new":
                    for i in range(len(self.tracker_instances) - 1):
                        if np.sum(ious[i][i+1:] > self.iou_threshold_tracker) > 0:
                            terminate.append(i)
                            # Replace id of new tracking instance with id of one to delete
                            j = 0
                            for k, iou in enumerate(ious[i][i+1:] > self.iou_threshold_tracker):
                                if iou == 1:
                                    j = k+i+1
                            self.tracker_instances[j].id = self.tracker_instances[i].id
                            predictions[j][4] = self.tracker_instances[i].id
                            if self.print_terminations:
                                print("Deleted - overlap - for the sake of new")
                elif self.mode == "old":
                    for i in sorted(range(len(self.tracker_instances)), reverse=True):
                        if np.sum(ious[i][:i] > self.iou_threshold_tracker) > 0:
                            terminate.append(i)
                            if self.print_terminations:
                                print("Deleted - overlap - for the sake of old")

                # Remove trackers with overlapping rois
                self._terminate_trackers(terminate)

                # Remove the predictions
                for i in sorted(terminate, reverse=True):
                    del predictions[i]

            # Check for overlap with detections (IF THERE ARE ANY ROIS)
            # and for novel rois (to start tracking)
            # and reconfirmation
            # Reformat the rois
            if rois:
                rois_ref = np.array(rois)
                rois_ref[:, 2:4] += rois_ref[:, 0:2]

                if predictions:
                    # Reformat the predictions
                    predictions_ref = np.array(predictions)
                    predictions_ref[:, 2:4] += predictions_ref[:, 0:2]

                    # Create an empty array for saving overlaps
                    overlap = np.zeros(shape=rois_ref.shape[0], dtype=int)

                    # Get the ious
                    for i, roi_ref in enumerate(rois_ref):
                        for j in range(len(predictions_ref)):
                            # Check for novel rois
                            iou = get_iou(roi_ref, predictions_ref[j])
                            if iou > self.iou_threshold_detector:
                                overlap[i] += 1
                                self.without_reconfirmation[j] = 0
                            else:
                                self.without_reconfirmation[j] += self.reconfirmation_counter_with_roi

                    # Save the novel rois
                    rois = np.array(rois)[overlap == 0]

            # Add reconfirmation when no rois
            else:
                for i in range(len(self.tracker_instances)):
                    self.without_reconfirmation[i] += self.reconfirmation_counter_without_roi

        # If there are detected rois that do not overlap with the predictions --> initialize new tracker
        for roi in rois:
            self._init_tracker(roi, frame)
            predictions.append(list(roi) + [0])     # Save the new rois as a prediction to return, add 0 as placeholder id

        # Return predictions
        return predictions

    def _init_tracker(self, bbox: list, frame: np.ndarray):
        """ Initializes a single KCF tracker """
        # Initialize the class
        self.tracker_instances.append(self.tracker_class(False, self.fixed_window, self.multiscale, self.occlusion_threshold))

        # Append empty list for predictions
        self.without_reconfirmation.append(0)

        # Start the tracking
        self.tracker_instances[-1].init(bbox, frame)

    def _terminate_trackers(self, list_trackers: list):
        """ Terminates list of trackers based on ids in the provided list """
        for i in sorted(list_trackers, reverse=True):
            del self.tracker_instances[i]
            del self.without_reconfirmation[i]


def get_iou(bb_test, bb_gt):
    """ Computes IOU between two bboxes in the form [x1,y1,x2,y2] """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)

    return o
