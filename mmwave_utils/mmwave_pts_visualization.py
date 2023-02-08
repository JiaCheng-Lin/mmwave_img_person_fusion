import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import json
import cv2

from io import StringIO, BytesIO
import PIL

from datetime import datetime 
import math

origin_px, origin_py = 6.0, 1.0 # jorjin mmwave UI

def process_json_data(data): # data: mmwave json
    detection = int(data["Detection"]) # number of person
    xy_list = []
    for i in range(detection):
        # print(data["JsonTargetList"][i]["ID"], data["JsonTargetList"][i]["Px"], data["JsonTargetList"][i]["Py"])

        # px = px*-1 <= flip horizontally because jorjinMMWave device app "display" part
        ID, px, py, Vx, Vy = data["JsonTargetList"][i]["ID"], \
                          round(data["JsonTargetList"][i]["Px"]-origin_px, 5),\
                          round(data["JsonTargetList"][i]["Py"]-origin_py, 5), \
                          data["JsonTargetList"][i]["Vx"], \
                          data["JsonTargetList"][i]["Vy"]
        # print("real:", px, py)
        xy_list.append([px, py, ID, (0, 0, 0), Vx, Vy]) # original pt: black color 
    return xy_list

# # generate a coordinate background for pts visualization.
def gen_background(coor_size):
    h, w, _ = coor_size # (600, 800, 3)
    
    # # using cv2 circle, line to draw coordinates.
    im = 255*np.ones(coor_size, dtype=np.uint8) # new an image, dtype is important
    origin_pt = (w//2, 30)
    line_color = (199, 195, 189)
    font_color = (94, 73, 52, 0.1)
    
     # # draw FOV line (120 degree)
    cv2.line(im, origin_pt, (0, int(origin_pt[0]/np.sqrt(3)+origin_pt[1])), (153, 187, 237)) # draw vertical axis line
    cv2.line(im, origin_pt, (w, int(origin_pt[0]/np.sqrt(3)+origin_pt[1])), (153, 187, 237)) # draw vertical axis line
    
    cv2.putText(im, "0", (origin_pt[0]+2, origin_pt[1]-3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, font_color, 1, cv2.LINE_AA)
    cv2.line(im, (origin_pt[0], 0), (origin_pt[0], h), line_color, 1) # draw y axis line
    cv2.line(im, (0, origin_pt[1]), (w, origin_pt[1]), line_color, 1) # draw x axis line
    cv2.circle(im, origin_pt, 5, (0, 0, 255), -1) # draw origin_pt

    # # draw coordinates lines, pts
    l, r = origin_pt[0], origin_pt[0]
    gap = 60 # each pts dis: 60 pixels
    cnt = 1
    while l>=0 and r<=w: # draw vertical axis, x axis pt
        l = origin_pt[0] + -1 * gap * cnt
        r = origin_pt[0] + gap * cnt

        cv2.line(im, (l, 0), (l, h), line_color) # draw vertical axis line
        cv2.line(im, (r, 0), (r, h), line_color) # draw vertical axis line
        cv2.putText(im, "-"+str(cnt), (l+2, origin_pt[1]-3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, font_color, 1, cv2.LINE_AA)
        cv2.putText(im, str(cnt), (r+2, origin_pt[1]-3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, font_color, 1, cv2.LINE_AA)

        cnt+=1

    y = origin_pt[1]
    cnt = 1
    while y<=h: # draw horizontal axis, y axis pt
        y = origin_pt[1] + gap * cnt

        cv2.line(im, (0, y), (w, y), line_color) # draw vertical axis line
        cv2.putText(im, str(cnt), (origin_pt[0]+2, y-3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, font_color, 1, cv2.LINE_AA)

        cnt+=1
   
    # cv2.imshow("bg", im)
    cv2.imwrite("mmwave_bg.png", im)
    # cv2.waitKey(0)
    

# # using opencv to draw, fast, spend about 0.001s per frame
def draw_mmwave_pts(bg, data={}, coor_size=(600, 800, 3), xy_list=[]): # data: mmwave json
    # start = datetime.now() ### time
    h, w, _ = coor_size # default coor_size: (600, 800, 3) 
    origin_pt = np.array((w//2, 30))
    gap = 60 # default pts dis: 60 pixels/meter

    # show origin points
    if len(xy_list) == 0 and data: # data(json file) exists.
        # bg = cv2.imread(r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\inference\mmwave_utils/mmwave_bg.png")
        xy_list = process_json_data(data) # [[px, py, ID], ... ], px, py: ... meter
        # print(xy_list)

        for px, py, ID, pt_color, Vx, Vy in xy_list:
            bg_pt = (origin_pt + (px*gap, py*gap)).astype(int)
            cv2.circle(bg, bg_pt, 4, pt_color, -1) # use the distinguish color from xy_list[3]

            # draw direction arrow
            Vx *= -1
            Vx, Vy = Vx/math.sqrt((Vx**2+Vy**2))*20, Vy/(math.sqrt(Vx**2+Vy**2))*20
            end_point = bg_pt + (int(Vx), int(Vy))
            # print(bg_pt, end_point)
            cv2.arrowedLine(bg, bg_pt, end_point, (0, 255, 0), 3)

            # write pos info
            dis = round(np.sqrt(px**2+py**2), 3)
            info = str(ID)+" ("+str(px)+", "+str(py)+") "+str(dis)
            cv2.putText(bg, info, (bg_pt[0]+5, bg_pt[1]-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (84, 153, 34), 1, cv2.LINE_AA)

            
    # show matched points
    else: 
        for px, py, ID, pt_color in xy_list:
            bg_pt = (origin_pt + (px*gap, py*gap)).astype(int)
            cv2.circle(bg, bg_pt, 4, pt_color, -1) # use the distinguish color from xy_list[3]
            
            dis = round(np.sqrt(px**2+py**2), 3)
            info = str(ID)+" ("+str(px)+", "+str(py)+") "+str(dis)
            cv2.putText(bg, info, (bg_pt[0]+5, bg_pt[1]-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (84, 153, 34), 1, cv2.LINE_AA)
    

    

    # cv2.imshow("mmwave_bg", bg)
    # cv2.waitKey(0)

    # end = datetime.now() 
    # t1 = (end-start).total_seconds()
    # print("t1", t1)

    return bg


# # using opencv to draw, fast, spend about 0.001s per frame
def draw_mmwave_pts_test(bg, data={}, coor_size=(600, 800, 3), xy_list=[]): # data: mmwave json
    # start = datetime.now() ### time
    h, w, _ = coor_size # default coor_size: (600, 800, 3) 
    origin_pt = np.array((w//2, 30))
    gap = 60 # default pts dis: 60 pixels/meter

    # show origin points
    if len(xy_list) == 0 and data: # data(json file) exists.
        # bg = cv2.imread(r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\inference\mmwave_utils/mmwave_bg.png")
        xy_list = process_json_data(data) # [[px, py, ID], ... ], px, py: ... meter
        # print(xy_list)

        for px, py, ID, pt_color, Vx, Vy in xy_list:
            bg_pt = (origin_pt + (px*gap, py*gap)).astype(int)
            cv2.circle(bg, bg_pt, 4, (255, 0, 255), -1) # use the distinguish color from xy_list[3]

            # draw direction arrow
            Vx *= -1
            Vx, Vy = Vx/math.sqrt((Vx**2+Vy**2))*20, Vy/(math.sqrt(Vx**2+Vy**2))*20
            end_point = bg_pt + (int(Vx), int(Vy))
            # print(bg_pt, end_point)
            cv2.arrowedLine(bg, bg_pt, end_point, (0, 255, 0), 3)

            # write pos info
            dis = round(np.sqrt(px**2+py**2), 3)
            info = str(ID)+" ("+str(px)+", "+str(py)+") "+str(dis)
            cv2.putText(bg, info, (bg_pt[0]+5, bg_pt[1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (84, 153, 34), 1, cv2.LINE_AA)

            
    # show matched points
    else: 
        for px, py, ID, pt_color in xy_list:
            bg_pt = (origin_pt + (px*gap, py*gap)).astype(int)
            cv2.circle(bg, bg_pt, 4, pt_color, -1) # use the distinguish color from xy_list[3]
            
            dis = round(np.sqrt(px**2+py**2), 3)
            info = str(ID)+" ("+str(px)+", "+str(py)+") "+str(dis)
            cv2.putText(bg, info, (bg_pt[0]+5, bg_pt[1]-2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (84, 153, 34), 1, cv2.LINE_AA)

    return bg


# # using plt to draw 
### Abandon, too slow to use in real-time, spend about 0.15s per frame
def plot_xy(xy_list):
    start = datetime.now() ### time

    # person xy
    x = [row[0] for row in xy_list] # or -> np.array(xy_list)[..., 0]
    y = [row[1] for row in xy_list]
    ID = [row[2] for row in xy_list]

    x_range = [-5, 6]
    y_range = [-1, 9]

    # FOV xy
    line_x_1 = np.linspace(-99, 0, 10)
    line_y_1 = line_x_1 / -np.sqrt(3) 
    line_x_2 = np.linspace(0, 99, 10)
    line_y_2 = line_x_2 / np.sqrt(3)

    plt.figure()
    plt.plot(0, 0, marker="o", markersize=8, color="red") # origin point
    plt.plot(x, y, 'o', markersize=5, color='blue')

    for i, label in enumerate(ID):
        dis = round(np.sqrt(x[i]**2+y[i]**2), 5)
        plt.text(x[i], y[i], str(label)+" ("+str(x[i])+", "+str(y[i])+") "+str(dis), color='green')

    # FOV lines
    plt.plot(line_x_1, line_y_1, '--', color='orange')
    plt.plot(line_x_2, line_y_2, '--', color='orange')


    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.xticks(range(x_range[0], x_range[1], 1))
    plt.yticks(range(y_range[0], y_range[1], 1))
    plt.grid()

    ax = plt.gca()#ax为上图
    ax.spines['right'].set_color('none')#删除右边缘黑框
    ax.spines['top'].set_color('none')#删除上边缘黑框
    ax.xaxis.set_ticks_position('bottom')#令x轴为底边缘
    ax.yaxis.set_ticks_position('left')#令y轴为左边缘
    ax.spines['bottom'].set_position(('data', 0))#将底边缘放到 y轴数据-1的位置
    ax.spines['left'].set_position(('data', 0))#将左边缘放到 y轴数据-1的位置

    ### time
    end = datetime.now() 
    t1 = (end-start).total_seconds()
    print("t1", t1)
    ### time

    # plt.savefig(json_path+json_name+'.png')
    # plt.show()
    # plt.close('all') # prevent error "Fail to allocate bitmap" when saving multiple images
    
    #### convert plt.show to cv2.imshow() -> ref: https://blog.csdn.net/zywvvd/article/details/109538750
    # BytesIO ref: https://blog.csdn.net/u013210620/article/details/79276280
    start = datetime.now() ### time

    buffer_ = BytesIO() # get buffer
    plt.savefig(buffer_, format='png') # save plt img to buffer(RAM)
    buffer_.seek(0) # move to "0" position
    dataPIL = PIL.Image.open(buffer_).convert("RGB") # read buffer img
    buffer_.close() # release buffer
    data = np.asarray(dataPIL)[:, :, ::-1]  # Convert RGB to BGR 
    
    ### time
    end = datetime.now() 
    t2 = (end-start).total_seconds()
    print("t2", t2)
    ### time

    # cv2.imshow('image', data)
    # cv2.waitKey(0)
    
    return data
    
    

if __name__ == "__main__":
    # json data
    data = {"TimeStamp": "09:44:02:180", "Detection": "2", \
            "JsonTargetList": [{"ID": 0, "Px": 6.2, "Py": 1.69, "Vx": 0.81, "Vy": 0.77, "Vrel": 0.79, "Ax": 1.14, "Ay": -0.02, "Arel": 0}, \
                            {"ID": 1, "Px": 9.91, "Py": 3.82, "Vx": -0.88, "Vy": -0.84, "Vrel": 0, "Ax": -1.17, "Ay": -0.01, "Arel": 0}]}

    # # get data
    # xy_list = process_json_data(data)
    # data = plot_xy(xy_list) # this way too slow to use (plt save)

    # # new way
    # gen_background(coor_size=(600, 800, 3))
    bg = cv2.imread(r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\inference\mmwave_utils/mmwave_bg.png")
    mmwave_visual = draw_mmwave_pts(bg, data, coor_size=(600, 800, 3)) # draw by opencv


