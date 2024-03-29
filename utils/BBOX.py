import numpy as np
import cv2  
import math

## Define a person bbox class
class BBOX(object):
    def __init__(self, tlwh, ID=None):
        
        self.w, self.h = int(tlwh[2]), int(tlwh[3])
        self.Xmin, self.Ymin = int(tlwh[0]), int(tlwh[1]),  # top left (x, y)
        self.Xmax, self.Ymax = int(self.Xmin+self.w), int(self.Ymin+self.h) # bottom right 
        self.center_x, self.center_y = self.Xmin+self.w//2, self.Ymin+self.h//2 # center point
        self.bottom_x, self.bottom_y = self.center_x, self.Ymax # bottom point
        
        self.ID = int(ID)

        self.Xr = None # estimated x (meter)
        self.Yr = None # estimated y (meter)
        self.Dis = None  # l2 norm(Xr, Yr)    
        self.bg_pt = None # for drawing in radar image  

        self.Dir = None # direction of bbox
        self.Still = None # people still or moving
        self.StillThreshold = 5 # pixel 

        self.MMW_ID = None # corresponding MMW ID
        self.matched_MMW = None # corresponding MMW()
        
        self.UID = None
    
    def addEstimatedXrYr(self, Xr, Yr, coor_size=(600, 800, 3)):
        self.Xr, self.Yr = Xr, Yr
        self.Dis = np.linalg.norm(np.array([self.Xr, self.Yr])) # l2 norm
        
        # for radar plane vis
        h, w, _ = coor_size
        origin_pt = np.array((w//2, h-30)) # (0, 0) in radar plane
        gap = 60 # default pts dis: 60 pixels/meter
        self.bg_pt = (origin_pt + (-self.Xr*gap, -self.Yr*gap)).astype(int)

    # add previous BBOX() infomation, like direction / if Still /...
    def addPreBBOXs(self, pre):
        Vx, Vy = self.center_x-pre.center_x, self.center_y-pre.center_y
        length = np.linalg.norm(np.array([Vx, Vy])) # l2 norm
        if (Vx == 0 and Vy == 0) or length <= self.StillThreshold: # people still -> bbox moves very little
            self.Still = True
            self.Dir = None
        else:
            Vx, Vy = Vx/length*20, Vy/length*20  # normalize the length in image
            self.Dir = (int(Vx), int(Vy))
            self.Still = False

    # draw center point in image
    def drawInCamera(self, img, pt_color=(0, 0, 255), pt_size=5, \
                    text="", text_size=0.6, text_color=(255, 255, 0), showArrow=False):
        
        # draw center point
        cv2.circle(img, (self.center_x, self.center_y), pt_size, pt_color, -1)

        # draw direction
        if self.Dir and showArrow:
            center_pt = np.array((self.center_x, self.center_y))
            end_point = center_pt + self.Dir
            cv2.arrowedLine(img, center_pt, end_point, (16, 137, 214), 3) 

        return img

    # draw estimated radar point in radar image(plane)
    def drawInRadar(self, bg_img, pt_color=(0, 0, 255), pt_size=4, \
                         showArrow=False):
        
        cv2.circle(bg_img, self.bg_pt, pt_size, pt_color, -1) 
        
        # draw text
        info = "CID: " + str(self.ID) # +" ("+str(round(-self.Xr, 2))+", "+str(round(self.Yr, 2))+") "
        cv2.putText(bg_img, info, (self.bg_pt[0]+5, self.bg_pt[1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (84, 153, 34), 1, cv2.LINE_AA)

        # draw arrow line
        if self.Dir and showArrow:
            end_point = self.bg_pt + self.Dir
            cv2.arrowedLine(bg_img, self.bg_pt, end_point, (16, 137, 214), 2)

        return bg_img

    def drawCorrespondingMID(self, img):
        cv2.putText(img, "- MID:"+str(self.MMW_ID)+" ", (self.Xmin+20, self.Ymin), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
        
        return img

    def drawUIDInRadar(self, bg_img, uid_trace, pt_size=5, unmatch_color=(0, 143, 255), match_pt_color=(232, 229, 26)):
        
        if self.UID: # not None
            # bg_img  = uid_trace.draw(bg_img, self.UID) # draw trace pt

            if self.MMW_ID != None and self.ID != None :
                pt_color = match_pt_color
                pt = self.matched_MMW.bg_pt
            else :
                pt_color = unmatch_color
                pt = self.bg_pt

            cv2.circle(bg_img, pt, pt_size, pt_color, -1)  # draw bg_pt point
            info = str(self.UID) + " " + str(self.MMW_ID) + " " + str(self.ID) 
            cv2.putText(bg_img, info, (pt[0]+5, pt[1]+5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (84, 153, 34), 1, cv2.LINE_AA)
            
           
            uid_trace.addTrace(self.UID, pt) # add pt to trace

        return bg_img, self.UID, uid_trace

    def drawBBOXInCamera(self, img, unmatch_color=(0, 143, 255), match_color=(232, 229, 26)):
        if self.UID: # not None
            if self.MMW_ID != None and self.ID != None :
                color = match_color
            else :
                color = unmatch_color


            cv2.rectangle(img, (self.Xmin, self.Ymin), (self.Xmax, self.Ymax), color=color, thickness=2) # bbox

            info = str(self.UID) + "-" + str(self.MMW_ID) + "-" + str(self.ID)
            cv2.putText(img, info, (self.Xmin, self.Ymin), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.1, color, 1, cv2.LINE_AA)
        
        return img



def list2BBOXCls(online_ids, online_tlwhs, pre_BBOXs):
    BBOXs = [] # save all BBOX() info
    for idx, tlwh in enumerate(online_tlwhs): # cur
        cur = BBOX(tlwh, online_ids[idx])
        for pre in pre_BBOXs: # pre BBOX()
            if cur.ID == pre.ID: 
                cur.addPreBBOXs(pre) # add pre info into cur BBOX(), like direction, ...
                pre_BBOXs.remove(pre)
                break

        BBOXs.append(cur)

    return BBOXs

