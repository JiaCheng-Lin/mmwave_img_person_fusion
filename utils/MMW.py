import numpy as np
import cv2  
import math

## Define a mmwave radar point class
class MMW(object):
    def __init__(self, Px, Py, ID=None, Vx=None, Vy=None, \
                Ax=None, Ay=None, Xc=None, Yc=None): # Xc, Yc: estimated x, y (pixel)
        self.Px = Px
        self.Py = Py

        self.ID = int(ID)
 
        self.Vx = Vx
        self.Vy = Vy
        self.Ax = Ax
        self.Ay = Ay

        self.Xc = Xc # estimated x (pixel)
        self.Yc = Yc # estimated y (pixel)

        self.Dir = None # direction of Vx, Vy
        if self.Vx and self.Vy:
            tmp_Vx, tmp_Vy = -self.Vx, -self.Vy # It is opposite to the data direction given by the app 
            length = np.linalg.norm(np.array([tmp_Vx, tmp_Vy]))
            tmp_Vx, tmp_Vy = tmp_Vx/length*20, tmp_Vy/length*20 # unit vector
            self.Dir = (int(tmp_Vx), int(tmp_Vy))

        self.BBOX_ID = None # corresponding BBOX ID

    # draw radar point in radar image(plane)
    def drawInRadar(self, bg_img, pt_color=(0, 0, 255), pt_size=4, coor_size=(600, 800, 3), showArrow=False):
        h, w, _ = coor_size
        origin_pt = np.array((w//2, h-30))
        gap = 60 # default pts dis: 60 pixels/meter
        
        # draw point
        bg_pt = (origin_pt + (-self.Px*gap, -self.Py*gap)).astype(int)
        cv2.circle(bg_img, bg_pt, pt_size, pt_color, -1) 
        
        # draw text
        dis = round(np.sqrt(self.Px**2+self.Py**2), 3)
        info = str(self.ID)  +" ("+str(round(-self.Px, 2))+", "+str(round(self.Py, 2))+") "#+str(dis)
        cv2.putText(bg_img, info, (bg_pt[0]+5, bg_pt[1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (84, 153, 34), 1, cv2.LINE_AA)
        
        # draw arrow line
        if self.Dir and showArrow:
            end_point = bg_pt + self.Dir
            cv2.arrowedLine(bg_img, bg_pt, end_point, (0, 255, 0), 2)
        
        return bg_img
    
    # draw estimated pixel point(Xc, Yc) in camera image
    def drawInCamera(self, img, pt_color=(0, 0, 255), pt_size=5, \
                    text="", text_size=0.8, text_color=(255, 255, 0), showArrow=False):

        # draw a estimated pixel point
        cv2.circle(img, (self.Xc, self.Yc), 2, pt_color, 5) 

        # draw text (MMWave ID and text)
        info = "MID: "+str(self.ID) + " " + text
        cv2.putText(img, info, (self.Xc, self.Yc+5), \
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, text_size, text_color, 1, cv2.LINE_AA)

        # draw arrow line
        if self.Dir and showArrow:
            bg_pt = np.array((self.Xc, self.Yc))
            end_point = bg_pt + self.Dir
            cv2.arrowedLine(img, bg_pt, end_point, (0, 255, 0), 2)

        return img


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

