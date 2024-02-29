"""
This is an implementation of a video motion detection method inspired by the paper written by Wei and Peng (2018).
This method is implemented only for single channel videos.
Note: this method is not an exact implementation of the paper by Wei and Peng (2018).

Authors: Hamideh Ayati, Vojtech Vlcek, Jenish Thapa, Prasil Adhikari, Philipp Zwetkoff

References:
    Wei, H. and Peng, Q., 2018. A block-wise frame difference method for real-time video motion detection.
    International Journal of Advanced Robotic Systems, 15(4), p.1729881418783633.
"""
import cv2 as cv
import numpy as np


# --------------- Detector Class ---------------
class PSNRMotionDetector:
    """ A class for motion detector """

    def __init__(self, window_size: int, stride: int, threshold_psnr: float, threshold_act: float, group_size: int,
                 sample_factor: int, bb_min_width: int, bb_min_height: int, rois_max_diff: int):
        """
        Init of the detector class.

        Parameters:
            window_size: size of a window for calculating the PSNR
            stride: specifies the movement by the sliding window (stride = window_size leads to no overlap)
            threshold_psnr: PSNR threshold for detecting motion
            threshold_act: threshold for pixel-level movement detection based on "active" patch memberships
            group_size: number of frames that should be added together
            sample_factor: specifies to which degree an image is blurred and down-sampled
            bb_min_width: minimum width of a bounding box
            bb_min_height: minimum height of a bounding box
            rois_max_diff: maximum amount of newly proposed regions of interest compared to the last proposal
        """
        # Hyperparameters
        self.window_size = window_size
        self.stride = stride
        self.threshold_psnr = threshold_psnr
        self.threshold_act = threshold_act
        self.group_size = group_size
        self.sample_factor = sample_factor
        self.bb_min_width = bb_min_width
        self.bb_min_height = bb_min_height
        self.rois_max_diff = rois_max_diff

        # Indices view for motion detection
        self.indices_view = None

        # Buffer grouping
        self.buffer_counter = 0
        self.buffer_curr = None
        self.buffer_past = None

        self.buffer_rois = None

        # Buffer rios
        self.count_past_rois = 0

    def __call__(self, frame):
        """ When the class instance is called """
        # Detect motion and propose regions of interest
        return self._rois_proposal(frame)

    def _rois_proposal(self, frame):
        """ Method for proposing regions of interest (ROIs) """
        # Down-sample the frame and change its type
        # (only one channel in our scenario necessary
        frame = self._down_sampling(frame[:, :, 0]).astype("float32")

        # Gather the information from the first frame (used for psnr method)
        if self.indices_view is None:
            self.indices_view = self._sig2col(
                np.arange(0, frame.shape[-2] * frame.shape[-1], 1, dtype=int).reshape(frame.shape[-2:]),
                (self.window_size, self.window_size),
                self.stride
            )

        # ---- Grouping the frames ----
        # If the group is not yet full, add the frame to the buffer
        if self.buffer_counter < self.group_size:
            # if the current is None (first frame in a group)
            if self.buffer_curr is None:
                self.buffer_curr = frame
            # Else add the current frame to the group
            else:
                self.buffer_curr += frame

            self.buffer_counter += 1

        # If the current group is full
        if self.buffer_counter == self.group_size:
            # and if the past group is present --> Make new prediction
            if self.buffer_past is not None:
                # Predict
                detection = self._psnr_block_motion_detection(
                    self._min_max_scale(self.buffer_curr),
                    self._min_max_scale(self.buffer_past)
                )

                # Change the dtype and up-sample
                detection = self._up_sampling(detection.astype("uint8"))

                # Retrieve the ROIs
                self.buffer_rois = self._get_rois(detection)

            # Save the current group as the past one and start the counter again
            self.buffer_past = self.buffer_curr
            self.buffer_curr = None
            self.buffer_counter = 0

        return self.buffer_rois if self.buffer_rois is not None else []

    def _psnr_block_motion_detection(self, curr_frame: np.ndarray, past_frame: np.ndarray):
        """ Detects motion based on current and past frame using sliding window and PSNR """
        # Prepare an empty output array
        output_frame = np.zeros_like(curr_frame, dtype=int)

        # Calculate the PSNR
        psnr = self._calculate_psnr(
            self._sig2col(past_frame, (self.window_size, self.window_size), self.stride),
            self._sig2col(curr_frame, (self.window_size, self.window_size), self.stride)
        )

        # Calculate how many times was a pixel above the threshold
        above_threshold = np.unique(self.indices_view[psnr <= self.threshold_psnr], return_counts=True)

        # Assign the numbers
        output_frame.reshape(-1)[above_threshold[0]] = above_threshold[-1]

        # If stride and window size are the same, do not normalize
        if self.stride == self.window_size:
            return output_frame * 255
        # Else normalize the output and apply second threshold
        else:
            output_frame = output_frame / np.square(self.window_size / self.stride)
            output_frame[output_frame < self.threshold_act] = 0

            return output_frame * 255

    @staticmethod
    def _sig2col(x, w_shape, stride=1, dilation=1):
        """ Represent signal so that each 'column' represents the elements in a sliding window """
        w_shape = np.asarray(w_shape)
        x_shape1, x_shape2 = np.split(x.shape, [-len(w_shape)])
        kernel_shape = dilation * (w_shape - 1) + 1
        out_shape2 = (x_shape2 - kernel_shape) // stride + 1

        # sliding window view (inspired by http://github.com/numpy/numpy/pull/10771)
        x_si1, x_si2 = np.split(x.strides, [len(x_shape1)])
        v_strides = tuple(x_si1) + tuple(stride * x_si2) + tuple(dilation * x_si2)
        v_shape = tuple(x_shape1) + tuple(out_shape2) + tuple(w_shape)
        _x = np.lib.stride_tricks.as_strided(x, v_shape, v_strides, writeable=False)
        return _x

    @staticmethod
    def _calculate_psnr(past, curr):
        """ Calculates the PSNR between the past and the current frames """
        psnr = np.ones_like(past)
        psnr *= np.mean((past - curr) ** 2, axis=(-2, -1), keepdims=True)

        mask = psnr == 0

        psnr[mask] = 100
        psnr[~mask] = 20 * np.log10(255 / np.sqrt(psnr[~mask]))

        return psnr

    def _down_sampling(self, frame: np.ndarray):
        """ Down-samples a frame """
        adj_frame = frame.copy()

        for _ in range(self.sample_factor):
            adj_frame = cv.pyrDown(adj_frame, dstsize=(adj_frame.shape[1] // 2, adj_frame.shape[0] // 2))

        return adj_frame

    def _up_sampling(self, frame: np.ndarray):
        """ Down-samples a frame """
        adj_frame = frame.copy()

        for _ in range(self.sample_factor):
            adj_frame = cv.pyrUp(adj_frame, dstsize=(adj_frame.shape[1] * 2, adj_frame.shape[0] * 2))

        return adj_frame

    def _min_max_scale(self, frame: np.ndarray):
        """ Scales frames based on their maximum and minimum value"""
        return (frame / (255 * self.group_size)) * 255

    def _get_rois(self, frame: np.ndarray):
        """ Returns proposed bounding boxes """
        contours, _ = cv.findContours(image=frame, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)

        rois = []
        for c in contours:
            # Extract coordinates and size of the bounding box
            x, y, w, h = cv.boundingRect(c)

            # Check the size
            if w <= self.bb_min_width or h <= self.bb_min_height:
                continue

            # Save the roi (x, y, width, height)
            #
            rois.append([x, y, w, h])

        # Check the proposed amount
        if abs(len(rois) - self.count_past_rois) >= self.rois_max_diff:
            return []

        self.count_past_rois = len(rois)

        return rois
