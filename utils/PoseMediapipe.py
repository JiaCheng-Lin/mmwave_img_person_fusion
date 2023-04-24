import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import cv2
import numpy as np

from pose_utils import process_bbox
import dataset1 

def getPoseFromMediapipe(mp_pose, mp_drawing, online_im, online_tlwhs):
    img_list = []
    for i, tlwh in enumerate(online_tlwhs):
        x, y, w, h = tlwh
        xmin = np.max((0, x))
        ymin = np.max((0, y))
        xmax = np.min((640 - 1, xmin + np.max((0, w - 1))))
        ymax = np.min((480 - 1, ymin + np.max((0, h - 1))))

        with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
            MARGIN=0
            #Media pose prediction ,we are 
            results = pose.process(online_im[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:])

            #Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing 
            mp_drawing.draw_landmarks(online_im[int(ymin)+MARGIN:int(ymax)+MARGIN,int(xmin)+MARGIN:int(xmax)+MARGIN:], results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    ) 
            img_list.append(online_im[int(ymin):int(ymax),int(xmin):int(xmax):])
            # cv2_imshow(image)
    return online_im
    # for img in img_list:
    #     cv2.imshow("img", img)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break