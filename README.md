# Repository for the Computer Vision Project

## Proposed Method: Real-Time Unsupervised Approach 
This approach relies on motion detection based on the peak signal-to-noise ratio (hereinafter PSNR) and tracking by correlation filters with motion estimations. 
### Motion Detection Based on PSNR 
This motion detection method was inspired by a paper written by Wei and Peng (2018). This method was selected for the project for not needing any training dataset and thus not being limited to a certain type of objects, its robustness against noise (background motion), interpretability, wide range of adjustability and good real-time performance.

#### The proposed method can be decomposed into the following steps: 
1. Input frames are blurred and downsampled. 
2. $n$ frames are added together to smoothen background motion and increase detection speed. This sum of $n$ frames forms an image at time $t$. 
3. The range of the image is scaled to a range from 0 to 255. 
4. The image at time $t$ is compared to the image at time $t-1$ to detect regions with motion. This is done in the following way:

	a) Based on window size and stride the images at $t$ and $t-1$ are split into patches.
	
	b) For an image patch at time $t$ and corresponding image patch at $t-1$ a PSNR is computed. The PSNR is computed as 20 times the common logarithm (with base 10) of the ration between the maximum signal value (255) and the mean squared error between the two patches.
	
	c) Only the patches below a given PSNR threshold are kept.
	
	d) If window size and stride are equal (a pixel can be a member of only one patch), then pixels in the patches below the given PSNR threshold are set to 255 and the others to 0. If window size and stride are not equal (a pixel can be a member of more than one patch), then the share of memberships in the patches below the given PSNR threshold is calculated and only the pixels with the share above a given threshold are set to 255, the rest is set to 0.

5. The image with the pixels of interest is upsampled. 
6. Based on the pixels of interest, regions of interest are found. 
7. Regions of interest with height or width under a given minimum are ignored. 
8. Regions of interest for the image at time $t$ are proposed. 

#### A visualization of the above-described method:
<img src="/figures/psnr_motion_detection_vis.png">

#### Based on the above overview of the method, certain conclusions can be drawn: 
1. Motion detection and thus proposal of regions of interest starts after $2*n$ frames. 
2. The proposed regions of interest change every $n$ frames (after the first $2*n$ frames). 

#### This method comes with certain limitations: 
1. Non-moving targets cannot be detected due to detection being based purely on motion. 
2. If the motion of the background is as significant as the target’s or if the target’s motion is as insignificant as of the background, then depending on the hyperparameters either target’s motion will not be detected, or false positives will be produced. 
3. Size and the coverage of proposed regions of interest is not consistent. This produces challenges for the subsequent tracking algorithm. 

### Tracking by Correlation Filters with Motion Estimations 
The tracking algorithm is based on a paper written by Xuan et al. (2020). The authors combined a motion estimation algorithm with kernelized correlation filter to address scenarios when a target is temporarily occluded. The key parts of their proposed method are:
1. combination of a Kalman filter and motion trajectory averaging as a motion estimation method (hereinafter ME),
2. effective solution for mitigating the boundary box effect in the KCF tracker.

The first part of the ME method is a Kalman filter for estimating the position and the velocity of moving objects. This method is limited by the amount of data to converge. That is why the motion trajectory averaging (hereinafter MTA) was introduced. The MTA algorithm is based on the observation of an object’s behavior in a short period of time. It assumes the position of the object at the current frame can be estimated using the speed and the position of the object in the previous frame. The amount of observations determines which method should be used first. Before the Kalman filter converges, MTA will be used as the output of ME, after convergence, the Kalman filter’s result will be used.

To address the occlusion scenario, a threshold is introduced for comparison with the peak value. For the peak values lower than the threshold the object is considered occluded and the position estimation by the ME used as the position of the object. However, when the peak value of the response patch obtained by the KCF tracker is higher than the threshold, the KCF is used again to update the Kalman filter.

### Combining the motion detection and tracking 
The tracker works hand in hand with the object detection algorithm. The proposed regions of interest by the detector are either used to initialize new tracking instances or for reconfirmation of the existing ones. An extra algorithm had to be created to combine the two approaches and enable multi-object tracking which is more challenging than its counterpart. The algorithm can be summarized in the following steps: 
1. Regions of interest are proposed by the object detector and current tracker instances. 
2. Trackers’ predictions under certain peak value threshold are removed and their tracking instances terminated.  
3. Initialization of new trackers: if the regions of interest proposed by the detector do not overlap with regions of interest of some existing tracking instance above a given intersection over union (hereinafter IoU) threshold, then a new tracking instance based on the proposed region of interest by the detector is initialized. 
4. Reconfirmation of the existing trackers: if the regions of interest proposed by the detector overlap with regions of interest of some existing tracking instance above a given IoU threshold, then this instance has been reconfirmed. Tracking instances not reconfirmed for n number of frames are terminated. 
5. Overlap of the existing trackers: if regions of interest of multiple tracking instances overlap above a given IoU threshold, then either the older or newer tracking instances are terminated (until there is no overlap). 
6. Regions of interest of the newly initialized and not terminated existing tracking instances are returned. 

#### Benefits from combining the detector with the tracker: 
1. The tracker stabilizes position and size of proposed regions of interest. 
2. The tracker estimates motion in the occluded areas in which the detector does not function well. 

#### Limitations stemming from the combination of the two methods: 
1. As a result of lost tracking instances, more false positives are produced. 
2. Poor tracking performance when a tracker is initialized based on an inaccurate region of interest or a region of interest with a highly occluded target. Consequently, tracker can fail to follow a target in occluded areas or predict its motion (tracker can get fixated on the occluded background instead of the target).

### How to run the proposed method
#### Dependencies
Create a conda environment based on the yml file in the proposed method's folder.
```
$ conda env create -f environment.yml
$ conda activate cv_lab_c7 
```

#### Running the Real-Time Detection and Tracking
To run the method, use the terminal to run the main.py and specify the path to a video and a config file.
```
$ python main.py video_path config_path
```
Example:
```
$ python main.py video_014.mp4 config.json
```
In the config file, you can setup various parameters which were mentioned in the method's description above. For all the generated tracks in the test videos the same parameters were used. These are the following (provided in the config file):
```
{
	"detector": {
		"window_size": 8,
		"stride": 2,
		"threshold_psnr": 25,
		"threshold_act": 0.3,
		"group_size": 5
	},
	"tracker": {
		"fixed_window": 1,
    	"multiscale": 1,
    	"occlusion_threshold": 0.50,
    	"stopping_threshold": 0.22
	}
}
```
*group_size = amount of frames added together

## Alternative Method: Supervised Approach (YOLOV5 + Kalman Filter Tracking)

This approach relies on object detection using a trained YOLOV5 model and tracking with Kalman Filters. It is an alternative method that was explored by us, but it was not favored because it is a supervised approach and the model inference time is significant in the absence of dedicated GPU hardware.

### Object Detection using YOLOV5 
The training of the YOLOV5 model on a custom dataset was done by following the YOLOV5 repository of Ultralytics [3].
First, the dataset we were given was annotated using Label Studio, resulting in annotations for all videos in the dataset. Then, the frames and their annotations were transformed into the YOLO format. About four videos were omitted randomly for later use as test data.
A YOLOV5 model was trained for 150 epochs on 16,000 frames with their respective annotations.

The trained model we obtained has pretty good generalization on the test videos.

#### Limitations of detection through YOLOV5 model: 
1. The inference process is significantly slow in the absence of dedicated GPU hardware.
2. The approach is supervised, requiring the annotation of the video to create a custom dataset.
3. The dataset is small and less diverse, with a high risk of overfitting due to the presence of many images that are just multiple frames of a single video.

### Tracking using Kalman Filter

The tracker follows the approach Simple Online and Realtime Tracking proposed by Bewley et. al. (2016). 
It only takes the detection position into account and propagates it to the next frame modelling the velocity via Kalman filter, based on past motion.
In each frame new detections are associated to the predictions of existing tracking instances represented by the Kalman filters. The association metric is the IOU-distance of detection and tracker prediction bounding boxes. Unassociated detections create new tracking instances if they get reconfirmed during a probationary state. 
The method does a good job predicting with accurate, frequent detections, and can handle short-term occlusions surprisingly well.
It is fast and only has to processes the current frame which makes it well suited for realtime applications. 

#### Limitations of Tracking using Kalman Filter:
* It depends on good and frequent detections.
* Target reidentification after long occlusion is a problem: specifically direction or velocity changes of the target lead to the tracker not being able to reidentify it, because it only models target velocity and position. 
* The reference system for all the coordinates is the drone, which means that the drawn tracking paths are moved as the drone moves e.g. because of wind.

### Combining YOLOV5 detection and Kalman Filter Tracking

Tracking results of Kalman Filter tracking for YOLOV5 detections on 28 thermal videos can be found [here](https://drive.google.com/drive/folders/1GwhAvD4SuyjSnPltXH1kgxhnVOy7WXLs). Videos 16-19 were not in the training set of the YOLOV5 model, and should give an idea of the generalizability of this approach. In the videos a bounding box indicates where humans are detected or predicted to be and the past tracking path is shown in white. The number on top of the bounding box is the id of the tracking instance, and the color of the bounding box indicates the state of the tracking instance:
* The bounding box is green if it is a detection associated with the prediction of a tracking instance. 
* The bounding box is red if it is the prediction of a confirmed tracking instance that wasn't associated with any detection, so the object is assumed to be occluded. If the tracking instance doesn't get reconfirmed for a specified number of frames it will get deleted.
* The bounding box is blue if it is a detection that wasn't associated to any tracking instance. These only appear in single frames and are barely visible.

### How to run the alternative method (YOLO detector + Kalman Filter tracker)
To install required dependencies with conda run:
```
$ conda env create -f environment.yaml
$ conda activate cv_lab_22 
```
To track in a video code:
```
$ cd yolo_detector_kf_tracker 
$ python main.py --video_path path/to/video_x.mp4
```
#### Hyperparameters of Kalman Filter Tracker
In `Sort` in `main.py`:
* "max_age": Maximum number of frames to keep alive a track without associated detections.
* "min_hits": Minimum number of associated detections before track is initialised.
* "iou_threshold": Minimum IOU for match.

To turn on or off debug information, live video display and to save output as a video or not set `processor.live_tracking(save_video=True, display_live=False, debug=False)` accordingly in `main.py`.

## References
1. Wei, H. and Peng, Q., 2018. A block-wise frame difference method for real-time video motion detection. International Journal of Advanced Robotic Systems, 15(4), p.1729881418783633.
2.  S. Xuan, S. Li, M. Han, X. Wan, G. Xia, “Object Tracking in Satellite Videos by Improved Correlation Filters With Motion Estimations”, 2020 IEEE Transactions on Geoscience and Remote Sensing, pp. 1074-1086, doi: 10.1109/TGRS.2019.2943366
3. Ultralytics. 2020. YOLOV5. https://github.com/ultralytics/yolov5 (2022)
4. A. Bewley, Z. Ge, L. Ott, F. Ramos, and B. Upcroft, ‘Simple Online and Realtime Tracking’, arXiv [cs.CV], 01-Feb-2016.

