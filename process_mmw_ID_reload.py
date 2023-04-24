"""mmwave""" 
from socket import *
from datetime import datetime
import json
import copy
# import time
import numpy as np

import sys
sys.path.append('../')
from utils.mmwave import * # import mmwave utils (functions)
"""mmwave""" 

from MMW_sort import Sort

## save path for image & mmwave
folderName = "20230309_131649"  
abs_path = r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\img_mmwave_data/"
data_dir = abs_path+'{}'.format(folderName)
# # read mmwave background image
bg = cv2.imread(r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\inference\utils/mmwave_bg_1.png")

def main():
    mot_tracker = Sort() 
    for filename in os.listdir(data_dir):
        if filename.startswith("image_"):
            img_path = os.path.join(data_dir, filename)
            frame = cv2.imread(img_path)

            mmwave_filename = f"mmwave{filename[5:-4]}.json" # get mmwave name 
            mmwave_path = os.path.join(data_dir, mmwave_filename)
            with open(mmwave_path, 'r') as f: ## get mmwave data
                sync_mmwave_json = json.load(f)

            # mmw_data = get_each_frame_data(sync_mmwave_json)
            # print(mmw_data)
            
            ## mmwave pts visualization by OpenCV, time consume: about 1ms. 
            bg_copy = copy.deepcopy(bg)
            # mmwave_pt_visual = draw_mmwave_pts(bg_copy, ori_mmwave_json)
            mmwave_pt_visual = draw_mmwave_pts_sync(bg_copy, sync_mmwave_json)
            cv2.imshow("mmwave", mmwave_pt_visual) # show radar ground plane img

            cv2.imshow("frame", frame) # show camera image at the same time
            ch = cv2.waitKey(60)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

def get_each_frame_data(data): # data: mmwave json
    origin_px, origin_py = 6.0, 1.0 # jorjin mmwave UI
    detection = int(data["Detection"]) # number of person
    xy_list = []
    for i in range(detection):
        # px = px*-1 <= flip horizontally because jorjinMMWave device app "display" part
        ID, px, py, Vx, Vy = data["JsonTargetList"][i]["ID"], \
                          round(data["JsonTargetList"][i]["Px"]-origin_px, 5),\
                          round(data["JsonTargetList"][i]["Py"]-origin_py, 5), \
                          data["JsonTargetList"][i]["Vx"], \
                          data["JsonTargetList"][i]["Vy"]

        xy_list.append([px, py, Vx, Vy]) 

    return xy_list


def process_json_data(data): # data: mmwave json
    origin_px, origin_py = 6.0, 1.0 # jorjin mmwave UI
    detection = int(data["Detection"]) # number of person
    xy_list = []
    for i in range(detection):
        # px = px*-1 <= flip horizontally because jorjinMMWave device app "display" part
        ID, px, py, Vx, Vy = data["JsonTargetList"][i]["ID"], \
                          round(data["JsonTargetList"][i]["Px"]-origin_px, 5),\
                          round(data["JsonTargetList"][i]["Py"]-origin_py, 5), \
                          data["JsonTargetList"][i]["Vx"], \
                          data["JsonTargetList"][i]["Vy"]
        # print("real:", px, py)
        # if ID == 0:
        xy_list.append([px, py, ID, (0, 0, 0), Vx, Vy]) # original pt: black color 
    return xy_list


# # using opencv to draw, fast, spend about 0.001s per frame
def draw_mmwave_pts_sync(bg, data={}, coor_size=(600, 800, 3), xy_list=[]): # data: mmwave json
    # start = datetime.now() ### time
    h, w, _ = coor_size # default coor_size: (600, 800, 3) 
    origin_pt = np.array((w//2, h-30))
    gap = 60 # default pts dis: 60 pixels/meter
    xy_list = process_json_data(data) # [[px, py, ID], ... ], px, py: ... meter

    for px, py, ID, pt_color, Vx, Vy in xy_list:
        bg_pt = (origin_pt + (-px*gap, -py*gap)).astype(int)
        cv2.circle(bg, bg_pt, 4, (255, 0, 255), -1) # use the distinguish color from xy_list[3]

        # draw direction arrow
        Vx, Vy = -Vx, -Vy # It is opposite to the data direction given by the app 
        Vx, Vy = Vx/math.sqrt((Vx**2+Vy**2))*20, Vy/(math.sqrt(Vx**2+Vy**2))*20
        end_point = bg_pt + (int(Vx), int(Vy))
        # print(bg_pt, end_point)
        # cv2.arrowedLine(bg, bg_pt, end_point, (0, 255, 0), 3)

        # write pos info
        dis = round(np.sqrt(px**2+py**2), 3)
        info = str(ID)+" ("+str(-px)+", "+str(py)+") "+str(dis)
        cv2.putText(bg, info, (bg_pt[0]+5, bg_pt[1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (84, 153, 34), 1, cv2.LINE_AA)

    return bg


if __name__ == "__main__":
    main()