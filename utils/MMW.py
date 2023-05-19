import numpy as np
import cv2  
import math

## Define a mmwave radar point class
class MMW(object):
    def __init__(self, Px, Py, ID="", Vx=None, Vy=None, \
                Ax=None, Ay=None, Xc=None, Yc=None): # Xc, Yc: estimated x, y (pixel)
        self.Px = Px
        self.Py = Py
        if ID != "":
            self.ID = int(ID)
        else:
            self.ID = ""
        self.Vx = Vx
        self.Vy = Vy
        self.Ax = Ax
        self.Ay = Ay
        self.Xc = Xc # estimated x 
        self.Yc = Yc # estimated y 

    def draw(self, bg_img, pt_color=(0, 0, 255), pt_size=4, coor_size=(600, 800, 3)):
        h, w, _ = coor_size
        origin_pt = np.array((w//2, h-30))
        gap = 60 # default pts dis: 60 pixels/meter
        
        bg_pt = (origin_pt + (-self.Px*gap, -self.Py*gap)).astype(int)
        cv2.circle(bg_img, bg_pt, pt_size, pt_color, -1) 

        dis = round(np.sqrt(self.Px**2+self.Py**2), 3)
        info = str(self.ID) # +" ("+str(round(-self.Px, 2))+", "+str(round(self.Py, 2))+") "#+str(dis)
        cv2.putText(bg_img, info, (bg_pt[0]+5, bg_pt[1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (84, 153, 34), 1, cv2.LINE_AA)
        
        if self.Vx and self.Vy:
            tmp_Vx, tmp_Vy = -self.Vx, -self.Vy # It is opposite to the data direction given by the app 
            tmp_Vx, tmp_Vy = tmp_Vx/math.sqrt((tmp_Vx**2+tmp_Vy**2))*20, tmp_Vy/(math.sqrt(tmp_Vx**2+tmp_Vy**2))*20
            end_point = bg_pt + (int(tmp_Vx), int(tmp_Vy))
            # print(bg_pt, end_point)
            cv2.arrowedLine(bg_img, bg_pt, end_point, (0, 255, 0), 2)
        
        return bg_img


# Convert <mmwave Json file> to <list contains MMW() class.
def json2MMWCls(MMW_json, origin_px=6.0, origin_py=1.0): # # origin_px/py: jorjin Device original point 
    MMWs = [] # save all mmwave radar points
    detection = int(MMW_json["Detection"]) # number of person
    for i in range(detection):
        ID, Px, Py, Vx, Vy, Ax, Ay  =   MMW_json["JsonTargetList"][i]["ID"], \
                                round(MMW_json["JsonTargetList"][i]["Px"]-origin_px, 5), \
                                round(MMW_json["JsonTargetList"][i]["Py"]-origin_py, 5), \
                                MMW_json["JsonTargetList"][i]["Vx"], \
                                MMW_json["JsonTargetList"][i]["Vy"], \
                                MMW_json["JsonTargetList"][i]["Ax"], \
                                MMW_json["JsonTargetList"][i]["Ay"]
        MMWs.append(MMW(Px, Py, ID, Vx, Vy, Ax, Ay))

    return MMWs


if __name__ == "__main__":
    a = MMW(1, 2, 2)

    print(a.ID)
    print(a.Px, a.Py)
    print(a.Vx)

