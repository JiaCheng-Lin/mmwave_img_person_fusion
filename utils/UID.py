import numpy as np
import cv2  
import math

# Define a UID class, it records the relationship between MMW() and BBOX()
class UID(object):
    def __init__(self, UID=None):
        self.UID = UID
        self.C_ID = None
        self.M_ID = None

# if MMW_ID is not exist in MMWs list -> delete the relationship {MID: UID} if exist before.
# because the MMW_ID is complementary
def filter_MMWs_UID(MMWs, MMWs_UID):
    MMWs_ID = []
    for mmw_cls in MMWs:
        MMWs_ID.append(mmw_cls.ID)
    
    new_MMWs_UID = {k: v for k, v in MMWs_UID.items() if k in MMWs_ID}

    return new_MMWs_UID

def checkUIDExists(BBOXs, MMWs, BBOXs_UID, MMWs_UID):
    for bbox_cls in BBOXs:
        if BBOXs_UID.get(bbox_cls.ID): 
            bbox_cls.UID = BBOXs_UID[bbox_cls.ID]
    
    for mmw_cls in MMWs:
        if MMWs_UID.get(mmw_cls.ID) and mmw_cls.OutOfImg: 
            mmw_cls.UID = MMWs_UID[mmw_cls.ID]

    return BBOXs, MMWs

def processMatch(MMWs, BBOXs, matches_idx_list, BBOXs_UID, MMWs_UID):
    for m_idx, b_idx in matches_idx_list:

        # # if MMW() has UID -> than match with no_UID_BBOX() -> BBOX get this UID.
        if MMWs[m_idx].UID and BBOXs[b_idx].UID==None:
            BBOXs[b_idx].UID = MMWs[m_idx].UID
            BBOXs_UID[BBOXs[b_idx].ID] = BBOXs[b_idx].UID 

        # # if MMW() in image and No UID, but BBOX() has UID
        if MMWs[m_idx].UID==None and BBOXs[b_idx].UID:
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