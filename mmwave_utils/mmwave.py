from socket import *
from datetime import datetime
import time
import json
import numpy as np
import cv2
import math
import lap # ref: https://github.com/gatagat/lap
# # lap.lapjv -> Jonker-Volgenant algorithm, faster than Hungarian Algorithm

## for regression model prediction. (mmwave pt project to image)
from joblib import dump, load # save and load model.
from sklearn.model_selection import train_test_split # split data to train&test
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures # Polynomial
from sklearn.pipeline import make_pipeline

import copy


## old version, abandon later, (20230209)
## Because of the change from single-threaded to dual-threaded
# Connect to the mmwave server and get data
def get_mmwave_data(frame_idx): 
    ### mmwave parameters ###
    
    host, port ="0.0.0.0", 6000
    addr = (host, port)
    buf = 2048
    if frame_idx == 1: # prevent LAG from (img processing)model initialization, so connect mmwave at frame 1.            
        print("Connect to the mmwave server...")
        global mmwave_socket
        mmwave_socket = socket(AF_INET, SOCK_DGRAM) # UDP # # reference: https://ithelp.ithome.com.tw/articles/10205819
        mmwave_socket.bind(addr)
    
    # start = datetime.now()
    data, addr = mmwave_socket.recvfrom(buf) # receive data from mmwave
    mmwave_json = json.loads(data) # to json format
    # end = datetime.now() 
    # print("t2 ", (end-start).total_seconds())
    return mmwave_json

def cal_time_error(t1, t2):
    # now_time = datetime.now()
    # current_time = now_time.time() # datetime.strptime(str(now_time), "%H%M%S%f")

    dT_now = datetime.combine(datetime.today(), t1) # combine the date 
    dT_mm = datetime.combine(datetime.today(), t2) # combine the date 
    
    time_error = (dT_now - dT_mm).total_seconds() # np.abs((dT_now - dT_mm).total_seconds()) # current_time-mmwave_time  # 0.015 second -> 15ms
    # print("Time Error:", time_error, "sec ", t1, t2)
    
    return time_error

def mmwave_extrapolation(mmwave_json, t):
    mtx_trans = np.array([[1, 0, t, 0, 0.5*t**2, 0],
                         [0, 1, 0, t, 0, 0.5*t**2],
                         [0, 0, 1, 0, t, 0],
                         [0, 0, 0, 1, 0, t],
                         [0, 0, 0, 0, 1, 0], 
                         [0, 0, 0, 0, 0, 1]]) # x, y, vx, vy, ax, ay

    detection = int(mmwave_json["Detection"]) # # number of person
    for i in range(detection): 
        x, y, vx, vy, ax, ay = mmwave_json["JsonTargetList"][i]["Px"], \
                               mmwave_json["JsonTargetList"][i]["Py"], \
                               -mmwave_json["JsonTargetList"][i]["Vx"], \
                               mmwave_json["JsonTargetList"][i]["Vy"], \
                               mmwave_json["JsonTargetList"][i]["Ax"], \
                               mmwave_json["JsonTargetList"][i]["Ay"]
        ori_mmwave = np.array([x, y, vx, vy, ax, ay])
        new_mmwave = np.matmul(mtx_trans, ori_mmwave)
        # print(ori_mmwave)
        # print(new_mmwave)
        
        mmwave_json["JsonTargetList"][i]["Px"], \
        mmwave_json["JsonTargetList"][i]["Py"], \
        mmwave_json["JsonTargetList"][i]["Vx"], \
        mmwave_json["JsonTargetList"][i]["Vy"], \
        mmwave_json["JsonTargetList"][i]["Ax"], \
        mmwave_json["JsonTargetList"][i]["Ay"] = new_mmwave

    return mmwave_json


## old version, abandon later, (20230209)
## Because of the change from single-thread to dual-thread
def mmwave_data_process(frame_idx, pre_mmwave_json):
    # """ get mmwave data"""
    now_img_time = datetime.now().time()
    # start = datetime.now()
    mmwave_json = get_mmwave_data(frame_idx) # Connect to the mmwave server and get data
    mmwave_json_time = datetime.strptime(mmwave_json['TimeStamp'], "%H:%M:%S:%f").time() # no date, so just convert to ".time()" format
    # end = datetime.now() 
    # t1 = (end-start).total_seconds()
    # print("t1", t1)
    # mmwave_json = {'TimeStamp': '19:29:13:287', 'Detection': '0', 'JsonTargetList': [{"ID": 0, "Px": 6.28, "Py": 1.59, "Vx": 0.05, "Vy": -0.04, "Vrel": 0, "Ax": -0.09, "Ay": 0, "Arel": 0}]}

    """ cal time error """ 
    time_error = cal_time_error(now_img_time, mmwave_json_time) # compare with image time(now time)

    ## radar fps, about 62ms
    if pre_mmwave_json:
        pre_mmwave_json_time = datetime.strptime(pre_mmwave_json['TimeStamp'], "%H:%M:%S:%f").time()
        mmwave_time_error = cal_time_error(mmwave_json_time, pre_mmwave_json_time)
        print("mmwave_time_error ", mmwave_time_error)

    
    """ Time Synchronization """
    sync_mmwave_json = copy.deepcopy(mmwave_json)
    ## extrapolation
    if time_error > 0:  # # Use Uniform Acceleration when T_cam > T_radar
        sync_mmwave_json = mmwave_extrapolation(sync_mmwave_json, time_error)

    ## interpolation
    elif time_error < 0 and pre_mmwave_json: # # Use Interpolation with Adjacent radar data when T_cam < T_radar
        
        print("1111111111111111")
        pass
    
    pre_mmwave_json = mmwave_json

    return time_error, sync_mmwave_json, mmwave_json, pre_mmwave_json


# mmwave data Time Synchronization
def mmwave_time_sync(mmwave_json, pre_mmwave_json, time_error):
    sync_mmwave_json = copy.deepcopy(mmwave_json)
    ## extrapolation
    if time_error > 0:  # # Use Uniform Acceleration when T_cam > T_radar
        sync_mmwave_json = mmwave_extrapolation(sync_mmwave_json, time_error)

    ## interpolation
    elif time_error < 0 and pre_mmwave_json: # # Use Interpolation with Adjacent radar data when T_cam < T_radar
        
        ## TODO, interpolation
        print("1111111111111111")
        pass
    
    return sync_mmwave_json


def pts_val_limit(x, y, w, h):
    x = -10 if x < 0 else x
    x = w+10 if x >= w else x
    y = -10 if y < 0 else y
    y = h+10 if y >= h else y

    return x, y

# project the mmwave pt(s) to image using transform "T"
def process_mmwave(mmwave_json, im0, origin_px=6.0, origin_py=1.0, regressor=None): # # origin_px/py: jorjin Device original point 
    T = np.array([[-168.79149551693234, 0.0724572081552246, 297.4314052813067], [-18.799447344663356, -97.81708310532088, 789.829879489633], [4.85722573273506e-17, -1.249000902703301e-16, 1.0000000000000013]])
    
    # # # Done: model initialization in mmwave_main file
    
    xy_list = [] # px, py, real_dis list in single frame
    detection = int(mmwave_json["Detection"]) # # number of person
    for i in range(detection): 

        # print("Vx, Vy: ", , mmwave_json["JsonTargetList"][i]["Vy"])

        # px = px*-1 <= flip horizontally because jorjinMMWave device app "display" part
        ID, px, py = mmwave_json["JsonTargetList"][i]["ID"], \
                    round(mmwave_json["JsonTargetList"][i]["Px"]-origin_px, 5), \
                    round(mmwave_json["JsonTargetList"][i]["Py"]-origin_py, 5) # minus the origin_x & y
        corresponding_uv = np.matmul(T,  np.array([px, py, 1]))
        corresponding_u, corresponding_v = int(corresponding_uv[0]/corresponding_uv[2]), int(corresponding_uv[1]/corresponding_uv[2])
        
        ### predict by sklearn, polynominal regression
        reg_uv = regressor.predict(np.array([[px, py]])) # origin 
        reg_u, reg_v = int(reg_uv[0][0]), int(reg_uv[0][1])

        ### RA regressor
        '''
        # regressor_RA = load(r'C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\cal_tranform_matrix\data/RA_data_2022_12_08_11_42_16.joblib') 
        dist = np.linalg.norm((px, py)) # l2 norm
        deg = 90-np.arctan2(py, px)*180.0/np.pi
        reg_uv_RA = regressor_RA.predict(np.array([[dist, deg]])) # use RA(range, angle) as input
        reg_u_RA, reg_v_RA = int(reg_uv_RA[0][0]), int(reg_uv_RA[0][1])
        '''

        ### convert by intrinsic parameters
        # camera_params = np.load("../camera_calibration/getK/intrinsic_parameters/camera_parameters_202211240103.npy", allow_pickle=True)[()]
        # mtx = np.array(camera_params['K'])
        # dist = np.array(camera_params['dist'])
        # points_2d = cv2.projectPoints(np.array([-px, -0.5, py]), np.array([0.0,0.0,0.0]), np.array([0.0,0.0, 0.0]), mtx, dist)[0]
        # # print(tuple(points_2d.flatten()))
        # a = points_2d.flatten()
        # print(a)
        # cv2.circle(im0, (int(a[0]), int(a[1])), 2, (0,255,255), 5) # 


        # print("transform", corresponding_u, corresponding_v)
        # print("regression", reg_u, reg_v)

        # # Done: pts_val_limit, avoid pts(u, v) value too large or small to "cv2.circle" error.
        # w, h, _ = im0.shape
        # reg_u, reg_v = pts_val_limit(reg_u, reg_v, w, h)
        # corresponding_u, corresponding_v = pts_val_limit(corresponding_u, corresponding_v, w, h)

        # # vis: draw the pts to image
        # cv2.circle(im0, (corresponding_u, corresponding_v), 2, (0, 89, 255), 5) # orange
        cv2.circle(im0, (reg_u, reg_v), 2, (255, 255, 0), 5) # light blue
        # cv2.circle(im0, (reg_u_RA, reg_v_RA), 2, (255, 0, 0), 5) # blue
        
        # # vis: show the "mmwave dis" at the estimated uv pos in img
        real_dis = round(np.sqrt(px**2+py**2), 5)
        cv2.putText(im0, str(real_dis), (reg_u, reg_v-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.8, (255, 255, 0), 1, cv2.LINE_AA)

        xy_list.append([reg_u, reg_v, real_dis, ID, px, py])

    ### original 
    # T = np.array([[-203.86088488354136, -19.672907486368345, 440.310396096428], [-3.1604213069951346, -70.84393142336387, 555.8573367228084], [-3.6991243401729434e-14, -7.563394355258879e-16, 1.0000000000000047]])
    # T = np.array([[-195.31193858179913, -14.788842727741137, 394.1668414663909], [-8.061606906490365, -62.775675859828034, 539.473064686669], [-3.0531133177191805e-16, -3.885780586188048e-16, 1.000000000000001]])
    # xy_list = [] # px, py, real_dis list in single frame
    # detection = int(mmwave_json["Detection"]) # # number of person
    # for i in range(detection): 
    #     # px = px*-1 <= flip horizontally because jorjinMMWave device app "display" part
    #     ID, px, py = mmwave_json["JsonTargetList"][i]["ID"], \
    #                 round(mmwave_json["JsonTargetList"][i]["Px"]-origin_px, 5), \
    #                 round(mmwave_json["JsonTargetList"][i]["Py"]-origin_py, 5) # minus the origin_x & y
    #     corresponding_uv = np.matmul(T,  np.array([px, py, 1]))
    #     corresponding_u, corresponding_v = int(corresponding_uv[0]/corresponding_uv[2]), int(corresponding_uv[1]/corresponding_uv[2])
    #     cv2.circle(im0, (corresponding_u, corresponding_v), 2, (255, 255, 0), 5)
        
    #     real_dis = round(np.sqrt(px**2+py**2), 5)
    #     cv2.putText(im0, str(real_dis), (int(corresponding_uv[0]), int(corresponding_uv[1])-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #         0.8, (255, 255, 0), 1, cv2.LINE_AA)

    #     xy_list.append([corresponding_u, corresponding_v, real_dis, ID, px, py])
    
    return im0, xy_list

### for sync
# project the mmwave pt(s) to image using transform "T"
def process_mmwave_sync(mmwave_json, im0, origin_px=6.0, origin_py=1.0, regressor=None): # # origin_px/py: jorjin Device original point 
    T = np.array([[-168.79149551693234, 0.0724572081552246, 297.4314052813067], [-18.799447344663356, -97.81708310532088, 789.829879489633], [4.85722573273506e-17, -1.249000902703301e-16, 1.0000000000000013]])
    
    # # # Done: model initialization in mmwave_main file
    
    xy_list = [] # px, py, real_dis list in single frame
    detection = int(mmwave_json["Detection"]) # # number of person
    for i in range(detection): 

        # print("Vx, Vy: ", , mmwave_json["JsonTargetList"][i]["Vy"])

        # px = px*-1 <= flip horizontally because jorjinMMWave device app "display" part
        ID, px, py, vx, vy = mmwave_json["JsonTargetList"][i]["ID"], \
                    round(mmwave_json["JsonTargetList"][i]["Px"]-origin_px, 5), \
                    round(mmwave_json["JsonTargetList"][i]["Py"]-origin_py, 5), \
                    mmwave_json["JsonTargetList"][i]["Vx"], \
                    mmwave_json["JsonTargetList"][i]["Vy"]
        
        ### predict by sklearn, polynominal regression
        reg_uv = regressor.predict(np.array([[px, py]])) # origin 
        reg_u, reg_v = int(reg_uv[0][0]), int(reg_uv[0][1])


        cv2.circle(im0, (reg_u, reg_v), 2, (0, 255, 0), 5) 
        # cv2.circle(im0, (reg_u_RA, reg_v_RA), 2, (255, 0, 0), 5) # blue
        
        # # vis: show the "mmwave dis" at the estimated uv pos in img
        real_dis = round(np.sqrt(px**2+py**2), 5)
        cv2.putText(im0, str(real_dis), (reg_u, reg_v+20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
            0.8, (0, 255, 0), 1, cv2.LINE_AA)

        xy_list.append([reg_u, reg_v, real_dis, ID, px, py, vx, vy])

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

def getErrorMatrix(xy_list, center_pt_list, error_threshold=200):
    # mmwave compare to image
    error_matrix = [] # two pairs of lists "error array"
    for xy_dis in xy_list: # mmwave
        px, py, real_dis, ID_mmwave, _, _, vx, vy = xy_dis
        vx, vy = -vx, -vy # rotate 180 degree, to align with the center_pt direction of image
        # print("vx, vy", vx, vy)
        row_error_matrix = []
        # print("xy_dis", xy_dis)
        for uv_dis in center_pt_list: # img center pts
            # print("uv_dis", uv_dis)
            u, v, fake_dis, ID_img, tlwh, dir_pt = uv_dis
            vec_dot = 1
            if dir_pt != (0, 0):
                img_vx, img_vy = dir_pt[0]-u, dir_pt[1]-v
                # print("img_vx, img_vy", img_vx, img_vy)
                # print("vx*vx, vy*vy", vx*img_vx, vy*img_vy)
                vec_dot = vx*img_vx+vy*img_vy
                # print("dot", vec_dot)

            error = math.sqrt((px-u)**2+(py-v)**2+((real_dis-fake_dis)*100)**2)
            # print("pts error", error)
            if not mmwavePts_within_bbox(px, py, tlwh) or error > error_threshold or vec_dot<=0 : # # return "True" if mmwave pt within bbox
                row_error_matrix.append(100000.0) # not match, the pt out of bbox
                continue
            # print("real_dis-fake_dis", real_dis-fake_dis)
            row_error_matrix.append(error)
        # print()
        error_matrix.append(row_error_matrix)
    
    return error_matrix # # row: mmwave, col: img(camera)

# # linear_assignment problem: Jonker-Volgenant algorithm, faster than Hungarian Algorithm
def linear_assignment(cost_matrix, thresh=10000.0): # thresh=1000.0: pt out of bbox, not match
    if cost_matrix.size == 0: # check whether the np.array is empty 
        return [], [], []
    matches, unmatched_a, unmatched_b = [], [], []
    c, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # print("lap.lapjv", c, x, y)
    
    for ix, mx in enumerate(x):# ix: mmwave idx, mx: img idx
        if mx >= 0: # matched！
            matches.append([ix, mx]) # [mmwave_idx, img_idx]
    unmatched_a = np.where(x < 0)[0] # mmwave unmatched pts ## np.where -> return idx
    unmatched_b = np.where(y < 0)[0] # camera/img unmatched pts
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

# # xy_list: mmwave list, [corresponding_u, corresponding_v, real_dis, ID, px, py]
# # center_pt_list: img list, [center_pt_x, center_pt_y, fake_dis, ID, tlwh]
def pt_match(xy_list, center_pt_list, im0, previous_ID_matches=[]):
    ## get error matrix between xy_list(mmwave) and center_pt_list(camera)
    error_matrix = getErrorMatrix(xy_list, center_pt_list, error_threshold=150)
    # print(error_matrix)

    # # linear_assignment problem, match the N x M matrix, ref: https://github.com/gatagat/lap
    # # get the minimum sum of weight, using lap.lapjv to solve it.
    matches, unmatched_mmwave, unmatched_img = linear_assignment(np.array(error_matrix))
    # print("matches", matches) # idx matches
    # print("unmatched_mmwave", unmatched_mmwave)
    # print("unmatched_img", unmatched_img)

    """ for Matched pts""" # the best situation
    new_mmwave_pts_list = []
    ID_matches = []
    for i, j in matches:
        # print("real_dis, fake_dis, diff",xy_list[i][2], center_pt_list[j][2], xy_list[i][2]-center_pt_list[j][2])

        ID_img = int(center_pt_list[j][3])
        l, t, _, _ = map(int, center_pt_list[j][4]) # map all para to int type
        _, _, real_dis, ID_mmwave, px, py, vx, vy = xy_list[i]
        
        cv2.putText(im0, "mmW "+str(real_dis), (l, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 0), 1, cv2.LINE_AA)

        new_mmwave_pts_list.append([px, py, ID_mmwave, (232, 229, 26)]) # give green color for vis to distinguish
        ID_matches.append([ID_mmwave, ID_img, px, py]) # save for previous ID matches

    # print("ID_matches", ID_matches)
    
    # print("previous_ID_matches", previous_ID_matches)


    """ unmatched problem: "Person Stopped", mmwave pt will disappear
         person in img and exists previous pos.""" 
    # # process the unmatched img person but matched before 

    # one/more person with no mmwave pos. (no mmwave data to match, so unmatch_img will be empty)
    if len(xy_list) == 0 and len(center_pt_list)!=0: 
        unmatched_img = [i for i in range(len(center_pt_list))]
    if len(unmatched_img)!=0: # unmatched_img idx
        for ID_mmwave, ID_img, px, py in previous_ID_matches:
            unmatched_img_ID = np.array(center_pt_list, dtype=object)[unmatched_img][:, 3]
            # print("ID_img", ID_img)
            # print("unmatched_img_ID", unmatched_img_ID)
            if ID_img in unmatched_img_ID:
                # print("find!!!!!!!!!!!!!!")
                ID_matches.append([ID_mmwave, ID_img, px, py]) # add to previous matches, record
                new_mmwave_pts_list.append([px, py, ID_mmwave, (0, 143, 255)]) # add to mmwave visualization
                # give orange color for vis to distinguish

                # # if find the previous ID matched, 
                # # so need to "delete" corresponding idx from "unmatched_mmwave"
                for i, idx in enumerate(unmatched_mmwave):
                    if ID_mmwave == xy_list[idx][3]:
                        unmatched_mmwave = np.delete(unmatched_mmwave, i)
                        break

    """camera can not capture person, but mmwave has pts""" # use mmwave original pt, just show it 
    # # solve the person out of view (camera can not capture, but mmwave can)
    if len(xy_list) != 0 and len(center_pt_list)==0:  # # no person 
        unmatched_mmwave = [i for i in range(len(xy_list))]
    # # if unmatched mmwave exists, just show it 
    if len(unmatched_mmwave)!=0: # unmatched_mmwave idx
        for idx in unmatched_mmwave:
            _, _, real_dis, ID_mmwave, px, py, vx, vy = xy_list[idx]
            new_mmwave_pts_list.append([px, py, ID_mmwave, (0, 0, 0)])  # add to mmwave visualization
            # give black color for vis to distinguish
    
    # print("new_mmwave_pts_list", new_mmwave_pts_list)

    return im0, new_mmwave_pts_list, ID_matches