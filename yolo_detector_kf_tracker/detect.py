import torch
import cv2
import json
import time
import numpy as np


class YOLO():
    """ CLASS FOR YOLOV5     """

    def __init__(self, model_path):
        #load yolov5 model trained on target tracking dataset
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = torch.hub.load('ultralytics/yolov5','custom', model_path).to(device) 
 


    def object_json(self, video_path, save_json= True, save_video = True):
            
        frame_num = 0
        video = cv2.VideoCapture(video_path)

        out = {}
        frames_list =[]
        while True:
            ret, frame = video.read()
            
            

            if not ret:
                 break
            

            results = self.model(frame)

    
            results = results.pandas().xyxy[0].to_json(orient="records")
            results = json.loads(results )
    

            print("Frame : ", frame_num)
            print(results)
            print('---------------------')

            if len(results) > 0:
                out[frame_num] = [] 
                for i in results:

                    #get coordinates and add them to json dict and draw rectangle on frame
                    xmin = int(round(int(i["xmin"]),0))
                    xmax = int(round(int(i["xmax"]),0))
                    ymin = int(round(int(i["ymin"]),0))
                    ymax = int(round(int(i["ymax"]),0))
                    out[frame_num].append(
                        {
                                "xmin" : xmin,
                                "ymin" : ymin,
                                "xmax" : xmax,
                                "ymax" : ymax
                        }
                        )
                    cv2.rectangle(frame, (xmin, ymin),(xmax, ymax), (0, 255, 0), 3)

            frames_list.append(frame)
            frame_num += 1
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


            
            video_name = video_path.split('/')[-1].split('.mp4')[0]

            #saves a json dict for objects coordinates
            if save_json:
                with open( video_name + ".json", "w") as out_file:
                    json.dump(out, out_file)

            #saves the result video where object are detected
            if save_video:
            
                frames_shape = np.array(frames_list)
            

                out = cv2.VideoWriter(video_name + "_det.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 30, (frames_shape.shape[2], frames_shape.shape[1]))

            for frame in frames_list:
                
                out.write(frame)

            out.release()

    


        video.release()
        cv2.destroyAllWindows()

    
    def __call__(self, frame):
        """ When the class instance is called """
        # Detect motion and propose regions of interest
        results = self.model(frame)
        results = results.pandas().xyxy[0].to_json(orient="records")
        results = json.loads(results)
        det_up =[]

        for i in results:
        

                #get coordinates and add them to json dict and draw rectangle on frame
                xmin = int(round((i["xmin"]),0))
                xmax = int(round(int(i["xmax"]),0))
                ymin = int(round(int(i["ymin"]),0))
                ymax = int(round(int(i["ymax"]),0))
                score = (round(float(i["confidence"]),2))
                det_up.append([xmin, ymin, xmax, ymax,score] )
        if len(results)<1:
            detections = np.ones((0,5))
            
        else:
            detections = np.array(det_up)
    
        return detections



# #load model
# yolo_model = YOLO("yolo_param.pt") #path for yolo parameters

# #load video
# yolo_model.object_json("data_target_tracking/video_019.mp4") #path for video; gives video and json as output
