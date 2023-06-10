import numpy as np
import cv2  
import math
from collections import deque
import json

## Define UID class, 
## record the trace of each UID.
class UID_Trace(object):
    def __init__(self):
        self.UID_trace = {}
        self.color_list = [ # trace color
            (128, 128, 0), (120, 218, 184), (222, 49, 99), (101, 136, 100), (206, 98, 214), 
            (128, 0, 0), (194, 255, 176), (241, 251, 111), (171, 223, 86), (255, 191, 0),
            (154, 115, 181), (255, 154, 118 ), (0, 159, 182), (117, 117, 117), (155, 88, 88),
            (189, 178, 255), (18, 250, 204), (97, 134, 23), (138, 96, 216)
        ]
        
        ### color visual test
        # img_white = 255*np.ones((500, 500, 3), np.uint8)
        # for i, color in enumerate(self.color_list):
        #     cv2.circle(img_white, (10+10*i, 10+10*i), 6, color, -1)
        # cv2.imshow("img_white", img_white)
        
    def addTrace(self, UID, trace_pt, trace_max_len=10):
        if self.UID_trace.get(UID): # find UID exists
            self.UID_trace[UID].append(trace_pt)
        else: # create new UID trace    
            self.UID_trace[UID] = deque(maxlen=trace_max_len)
            self.UID_trace[UID].append(trace_pt)
        # print(self.UID_trace[UID])

    def draw(self, bg_img, UID):
        if self.UID_trace.get(UID):
            for pt in self.UID_trace[UID]:
                cv2.circle(bg_img, pt, 3, self.color_list[UID % len(self.color_list)], -1)
            
        return bg_img

# if MMW_ID is not exist in MMWs list -> delete the relationship {MID: UID} if exist before.
# because the MMW_ID is complementary
def filter_MMWs_UID(MMWs, MMWs_UID):
    MMWs_ID = []
    for mmw_cls in MMWs:
        MMWs_ID.append(mmw_cls.ID)
    
    new_MMWs_UID = {k: v for k, v in MMWs_UID.items() if k in MMWs_ID}

    return new_MMWs_UID

# Check the <BBOXs> and <MMWs out of image> which have got UID
# And give the UID to current class().
def checkUIDExists(BBOXs, MMWs, BBOXs_UID, MMWs_UID):
    BBOXs_UID_SET = set() # prevent more than one bbox in image -> SAME UID
    for bbox_cls in BBOXs:
        cid = bbox_cls.ID
        if BBOXs_UID.get(cid):
            uid = BBOXs_UID[cid]
            if uid not in BBOXs_UID_SET: 
                BBOXs_UID_SET.add(uid)
                bbox_cls.UID = uid
            else:
                BBOXs_UID.pop(cid) # if find two bbox have same UID, than delate the second .
    
    MMWs_UID_SET = set() # prevent more than one MMW in image have SAME UID
    for mmw_cls in MMWs:
        mid = mmw_cls.ID
        if MMWs_UID.get(mid) and mmw_cls.OutOfImg: 
            uid = MMWs_UID[mid]
            if uid not in MMWs_UID_SET:
                MMWs_UID_SET.add(uid)
                mmw_cls.UID = uid
            else :
                MMWs_UID.pop(mid)

    return BBOXs, MMWs

def processMatch(MMWs, BBOXs, matches_idx_list, BBOXs_UID, MMWs_UID):
    for m_idx, b_idx in matches_idx_list:
        print(MMWs[m_idx].ID, ": ", MMWs[m_idx].UID )
        print(BBOXs[b_idx].ID, ": ", BBOXs[b_idx].UID,)

        # # if MMW() has UID -> than match with no_UID_BBOX() -> BBOX get this UID.
        if MMWs[m_idx].UID != None and BBOXs[b_idx].UID == None:
            BBOXs[b_idx].UID = MMWs[m_idx].UID
            BBOXs_UID[BBOXs[b_idx].ID] = BBOXs[b_idx].UID 

        # # if MMW() in image and No UID, but BBOX() has UID
        if MMWs[m_idx].UID == None and BBOXs[b_idx].UID != None:
            MMWs[m_idx].UID = BBOXs[b_idx].UID
            MMWs_UID[MMWs[m_idx].ID] = MMWs[m_idx].UID


        # # if MMW() has UID && BBOX() has UID -> match 
        # # -> change MMW()'s UID to BBOX()'s UID
        if MMWs[m_idx].UID and BBOXs[b_idx].UID and (MMWs[m_idx].UID != BBOXs[b_idx].UID):
            print("!!!!! both have UID but not same !!!!!")
            MMWs[m_idx].UID = BBOXs[b_idx].UID
            MMWs_UID[MMWs[m_idx].ID] = MMWs[m_idx].UID

        # print(MMWs[m_idx].UID, BBOXs[b_idx].UID)

    return MMWs, BBOXs, BBOXs_UID, MMWs_UID

def UID_assignment(MMWs, BBOXs, matches_idx_list, BBOXs_UID, MMWs_UID, UID_number):
    # print("UID_number", UID_number) 
    MMWs_UID = filter_MMWs_UID(MMWs, MMWs_UID) # remove <MMW_ID in MMWs_UID dict>(relationship) if MMW_ID is not exist in MMWs list.
    
    # Check the <BBOXs> and <MMWs out of image> 
    #      !!! which have got UID !!!
    BBOXs, MMWs = checkUIDExists(BBOXs, MMWs, BBOXs_UID, MMWs_UID)

    # when MMW() from Out to In
    MMWs, BBOXs, BBOXs_UID, MMWs_UID = processMatch(MMWs, BBOXs, matches_idx_list, BBOXs_UID, MMWs_UID)

    # give each BBOX a <UID>, 
    # and give each MMW a <UID> that estimated (Xc, Yc) not in the image.
    for bbox_cls in BBOXs:
        if not BBOXs_UID.get(bbox_cls.ID) and bbox_cls.UID==None:
            BBOXs_UID[bbox_cls.ID] = UID_number
            bbox_cls.UID = UID_number
            UID_number += 1

    for mmw_cls in MMWs:
        if not MMWs_UID.get(mmw_cls.ID) and mmw_cls.OutOfImg and mmw_cls.UID==None: # out of cam image
            MMWs_UID[mmw_cls.ID] = UID_number
            mmw_cls.UID = UID_number
            UID_number += 1

    return  BBOXs, MMWs, BBOXs_UID, MMWs_UID, UID_number