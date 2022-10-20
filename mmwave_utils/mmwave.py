from socket import *
from datetime import datetime
import time
import json
import numpy as np
import cv2
import math

# Connect to the mmwave server and get data
def get_mmwave_data(frame_idx): 
    ### mmwave parameters ###
    start = datetime.now()
    host, port ="0.0.0.0", 6000
    addr = (host, port)
    buf = 2048
    if frame_idx == 1: # prevent LAG from (img processing)model initialization, so connect mmwave at frame 1.            
        print("Connect to the mmwave server...")
        global mmwave_socket
        mmwave_socket = socket(AF_INET, SOCK_DGRAM) # UDP # # reference: https://ithelp.ithome.com.tw/articles/10205819
        mmwave_socket.bind(addr)
    
    data, addr = mmwave_socket.recvfrom(buf) # receive data from mmwave
    mmwave_json = json.loads(data) # to json format
    end = datetime.now() 
    # print((end-start).total_seconds())
    return mmwave_json

def cal_time_error(now_time, mmwave_json):
    # now_time = datetime.now()
    current_time = now_time.time() # datetime.strptime(str(now_time), "%H%M%S%f")

    mmwave_time = datetime.strptime(mmwave_json['TimeStamp'], "%H:%M:%S:%f").time() # no date, so just convert to ".time()" format
    dT_mm = datetime.combine(datetime.today(), mmwave_time) # combine the date 
    dT_now = datetime.combine(datetime.today(), current_time) # combine the date 
    time_error =  np.abs((dT_now - dT_mm).total_seconds()) #current_time-mmwave_time  # 0.015 second -> 15ms
    print("Time Error:", time_error, "sec ", current_time, mmwave_time)
    
    return time_error

def mmwave_data_process(frame_idx, mmwave_json):
    # """ get mmwave data"""
    now_img_time = datetime.now()
    start = datetime.now()
    mmwave_json = get_mmwave_data(frame_idx) # Connect to the mmwave server and get data
    end = datetime.now() 
    t1 = (end-start).total_seconds()
    # mmwave_json = {'TimeStamp': '19:29:13:287', 'Detection': '0', 'JsonTargetList': [{"ID": 0, "Px": 6.28, "Py": 1.59, "Vx": 0.05, "Vy": -0.04, "Vrel": 0, "Ax": -0.09, "Ay": 0, "Arel": 0}]}

    # """ cal time error """ 
    time_error = cal_time_error(now_img_time, mmwave_json) # compare with image time(now time)
    
    return time_error, mmwave_json
    
# project the mmwave pt(s) to image using transform "T"
def process_mmwave(mmwave_json, im0, origin_px=6.0, origin_py=1.0): # # origin_px/py: jorjin Device original point 
    T = np.array([[298.3232162910885, -5.776874682271578, 296.5718869465865], 
                 [-0.47510099657455684, -11.295472784043396, 309.37445188343264], 
                 [-3.831570477563773e-15, 2.8408265323465187e-15, 0.9999999999999895]])

    xy_list = [] # px, py, real_dis list in single frame
    detection = int(mmwave_json["Detection"]) # # number of person
    for i in range(detection): 
        # px = px*-1 <= flip horizontally because jorjinMMWave device app "display" part
        ID, px, py = mmwave_json["JsonTargetList"][i]["ID"], round(mmwave_json["JsonTargetList"][i]["Px"]-origin_px, 5)*-1, round(mmwave_json["JsonTargetList"][i]["Py"]-origin_py, 5) # minus the origin_x & y
        corresponding_uv = np.matmul(T,  np.array([px, py, 1]))
        corresponding_u, corresponding_v = int(corresponding_uv[0]/corresponding_uv[2]), int(corresponding_uv[1]/corresponding_uv[2])
        cv2.circle(im0, (corresponding_u, corresponding_v), 2, (255, 255, 0), 5)
        
        real_dis = round(np.sqrt(px**2+py**2), 5)
        cv2.putText(im0, str(real_dis), (int(corresponding_uv[0]), int(corresponding_uv[1])-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.8, (255, 255, 0), 1, cv2.LINE_AA)

        xy_list.append([corresponding_u, corresponding_v, real_dis, ID])
    
    return im0, xy_list

# return "True" if mmwave pt within bbox 
def mmwavePts_within_bbox(px, py, tlwh):
    left, right, top, down = tlwh[0], tlwh[0]+tlwh[2], tlwh[1], tlwh[1]+tlwh[3] # bbox edge range
    # print("mmwave pt", px, py)
    # print("bbox range", left, right, top, down)
    if px >= left and px <= right and py >= top and py <= down:
        # print("True")
        return True
    # print("False")
    return False

# xy_list: mmwave list, center_pt_list: img list
def pt_match(xy_list, center_pt_list, im0):
    # mmwave compare to image
    error_match = [] # two pairs of lists "error array"
    for xy_dis in xy_list: # mmwave
        px, py, real_dis, ID_mmwave = xy_dis
        row_error_match = []
        # print("xy_dis", xy_dis)
        for uv_dis in center_pt_list: # img center pts
            # print("uv_dis", uv_dis)
            u, v, fake_dis, ID_img, tlwh = uv_dis
            if not mmwavePts_within_bbox(px, py, tlwh): # # return "True" if mmwave pt within bbox
                row_error_match.append(100000.0) # not match
                continue
            error = math.sqrt((px-u)**2+(py-v)**2+((real_dis-fake_dis)*100)**2)
            # print("real_dis-fake_dis", real_dis-fake_dis)
            row_error_match.append(error)
        error_match.append(row_error_match)
    # print(error_match)

    pt_relation = []
    if error_match:
        print(error_match)
        row_mark = [] # will be pass after find the matched pt.
        for mm_idx in range(len(error_match[0])):
            min_r, min_c, min_error = -1, -1, 100000.0
            for im_idx in range(len(error_match)):
                if im_idx in row_mark: # pass after find the matched pt.
                    continue
                if min_error > error_match[im_idx][mm_idx]:
                    min_error = error_match[im_idx][mm_idx]
                    min_r, min_c = im_idx, mm_idx
                    # print(min_r, min_c)
            # print(min_r, min_r)
            if min_r != -1 and min_c != -1:
                row_mark.append(min_r)
                pt_relation.append([min_r, min_c])

        # print(pt_relation)
    for pt in pt_relation:
        i, j = pt[0], pt[1]
        cv2.putText(im0, "mmW "+str(xy_list[i][2]), (center_pt_list[j][0], center_pt_list[j][1]), 
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 0), 1, cv2.LINE_AA)

    return im0