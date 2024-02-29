"""
This code has been taken from https://github.com/SY-Xuan/CFME
It is an implementation of the paper by Xuean et al. (2020).
Parts of the codes were adjusted for our needs.

References:
    Xuan, S., Li, S., Han, M., Wan, X. and Xia, G.S., 2019.
    Object tracking in satellite videos by improved correlation filters with motion estimations.
    IEEE Transactions on Geoscience and Remote Sensing, 58(2), pp.1074-1086.
"""

from kcftracker import KCFTracker
import cv2
import numpy as np

class mecfTracker():
    count = 1
    def __init__(self, hog=False, fixed_window=True, scale=False, occlusion_threshold=0.3):
        self.tracker = KCFTracker(
            hog=hog,
            fixed_window=fixed_window,
            multiscale=scale,
            occlusion_threshold=occlusion_threshold
        )
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001
        self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
        self.trace_array = []
        self.predict = [0, 0]
        # whether kalman filter can be used
        self.iskalman_work = False
        # whether the object is occluded
        self.isocclution = False

        self.occlution_index = 0
        self.tem_trace_array = []
        self.frame_index = 0
        self.last_bbox = []
        self.kalman_work_num = 0
        self.occlusion_threshold = occlusion_threshold
        self.id = mecfTracker.count
        mecfTracker.count += 1

    def init(self, bbox, frame):
        self.trace_array.append(bbox)
        self.kalman.correct(np.array([[np.float32(bbox[0])],[np.float32(bbox[1])]]))
        self.tracker.init([bbox[0],bbox[1],bbox[2],bbox[3]], frame)
        self.frame_index += 1

    def update(self, frame):
        if self.iskalman_work:
            next_bbox = [self.predict[0], self.predict[1], self.last_bbox[2], self.last_bbox[3]]
            self.last_bbox, peak_value = self.tracker.update(frame, next_bbox, isUse=True)
            # long-term
            if peak_value > self.occlusion_threshold:
                self.trace_array.append(self.last_bbox.copy())
                self.kalman.correct(np.array([[np.float32(self.last_bbox[0])],[np.float32(self.last_bbox[1])]]))
                self.predict = self.kalman.predict()
                
            else:
                self.last_bbox = [next_bbox[0], next_bbox[1], self.last_bbox[2], self.last_bbox[3]]
                self.predict = self.kalman.predict()
        else:
            if len(self.trace_array) > 8:   # Original: 4
                dx = 0
                dy = 0
                for i in range(-5, -1):
                    dx += self.trace_array[i + 1][0] - self.trace_array[i][0]
                    dy += self.trace_array[i + 1][1] - self.trace_array[i][1]
                next_bbox = [self.last_bbox[0] + dx / 4, self.last_bbox[1] + dy / 4, self.last_bbox[2], self.last_bbox[3]]
                self.last_bbox, peak_value = self.tracker.update(frame, next_bbox, isUse=True)
                # long-term
                if peak_value < self.occlusion_threshold:
                    self.last_bbox = [next_bbox[0], next_bbox[1], self.last_bbox[2], self.last_bbox[3]]
                    self.isocclution = True
                else:
                    if self.isocclution == True:
                        if self.occlution_index != 0:
                            self.tem_trace_array.append(self.last_bbox.copy())
                        self.occlution_index += 1
                        
                        if self.occlution_index == 6:
                            
                            self.trace_array.extend(self.tem_trace_array)
                            self.isocclution = False
                    else:
                        self.trace_array.append(self.last_bbox.copy())
                
            else:
                self.last_bbox, peak_value = self.tracker.update(frame)
                self.trace_array.append(self.last_bbox.copy())

            # Check if the kalman is working
            # if (abs(self.predict[0] - self.last_bbox[0]) < 1) and (abs(self.predict[1] - self.last_bbox[1]) < 1):
            if get_iou(self.predict, self.last_bbox) > 0.1:
                self.kalman_work_num += 1
                if self.kalman_work_num == 3:
                    self.iskalman_work = True
                    self.kalman_work_num = 0
            else:
                self.kalman_work_num = 0
            self.kalman.correct(np.array([[np.float32(self.last_bbox[0])], [np.float32(self.last_bbox[1])]]))
            self.predict = self.kalman.predict()
        self.frame_index += 1
        roi = self.last_bbox + [self.id]
        return roi, peak_value


def get_iou(bb_test, bb_gt):
    """ Computes IOU between two bboxes in the form [x1,y1,x2,y2] """
    if len(bb_test) == 2:
        return 0

    bb_test = np.array(bb_test).flatten()
    bb_test[2:4] += bb_test[0:2]

    bb_gt = np.array(bb_gt)
    bb_gt[2:4] += bb_gt[0:2]

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