import numpy as np
import cv2  
import math
import lap # ref: https://github.com/gatagat/lap


# caluate the error between a bbox() and a MMW()
# pixel point / position(meter) error
def cal_BBOX_MMW_error(MMW, BBOX):
    center_pt, estimated_center_pt = np.array((BBOX.center_x, BBOX.center_y)), np.array((MMW.Xc, MMW.Yc))
    pos, estimated_pos = np.array((MMW.Px, MMW.Py)), np.array((BBOX.Xr, BBOX.Yr))

    pixel_error = np.linalg.norm(center_pt-estimated_center_pt) # bbox - mmw
    meter_error = np.linalg.norm(pos-estimated_pos) # mmw - bbox

    return pixel_error, meter_error

def cal_BBOX_MMW_error_reg(MMW, BBOX):
    center_pt, estimated_center_pt = np.array((BBOX.center_x, BBOX.center_y)), np.array((MMW.reg_Xc, MMW.reg_Yc))

    pixel_error = np.linalg.norm(center_pt-estimated_center_pt) # bbox - mmw

    return pixel_error

def cal_BBOX_MMW_error_T(MMW, BBOX):
    center_pt, estimated_center_pt = np.array((BBOX.center_x, BBOX.center_y)), np.array((MMW.T_Xc, MMW.T_Yc))

    pixel_error = np.linalg.norm(center_pt-estimated_center_pt) # bbox - mmw

    return pixel_error


# check if the estimated/projected MMW point(Xc, Yc) is in the bbox
def checkWithinBbox(MMW, BBOX):
    if BBOX.Xmin <= MMW.Xc <= BBOX.Xmax and BBOX.Ymin <= MMW.Yc <= BBOX.Ymax:
        return True
    else:
        return False

# assign MMW to BBOX, one to one
def linear_assignment(cost_matrix, thresh=10000.0):
    matches, unmatched_a, unmatched_b = [], [], []
    c, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # print("lap.lapjv", c, x, y)
    # print("cost_matrix", cost_matrix)
    
    for ix, mx in enumerate(x):# ix: mmw idx, mx: bbox idx
        if mx >= 0: # matched！
            matches.append([ix, mx]) # [[mmw_idx, bbox_idx], ...]
    unmatched_a = np.where(x < 0)[0] # mmw unmatched pts ## np.where -> return idx
    unmatched_b = np.where(y < 0)[0] # bbox unmatched pts
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

# direction check: if two dir degree>90(dot<0) -> unmatch
# cal the degree/dot of two dir
def sameDir(MMW_dir, BBOX_dir):
    if MMW_dir and BBOX_dir: # if two dir exists
        dir_dot = np.dot(np.array(MMW_dir), np.array(BBOX_dir))
        #  print("dot", dir_dot)  # print("degree", np.arccos(dir_dot/np.linalg.norm(np.array(mmw_cls.Dir))/np.linalg.norm(np.array(bbox_cls.Dir)))*180/np.pi)
        if dir_dot < 0: # degree > 90
            return False 
    return True # have not dir / dir_dot>=0 (degree<=90)

def CalErrorMatrix(MMWs, BBOXs, error_threshold=150, \
                   pixel_weight=1, meter_weight=200): #250
    unmatch_value = 1e6 
    error_mtx = [] # m x n
    
    for mmw_cls in MMWs:
        row_error_mtx = [] # save the ERRORs: one MMW() to each BBOX() 
        for bbox_cls in BBOXs:
            if checkWithinBbox(mmw_cls, bbox_cls) == True: # MMW() in bbox
                pixel_error, meter_error = cal_BBOX_MMW_error(mmw_cls, bbox_cls) # cal th err
                error = pixel_error*pixel_weight + meter_error*meter_weight # weight

                if error>error_threshold: # or sameDir(mmw_cls.Dir, bbox_cls.Dir)==False: 
                    row_error_mtx.append(unmatch_value) #  give a large number to <error mtx>
                else:
                    row_error_mtx.append(error) # add to error mtx
                    print(mmw_cls.ID, bbox_cls.ID, error)

            else: 
                row_error_mtx.append(unmatch_value)  # not in the bbox -> give a large number to <error mtx>

        error_mtx.append(row_error_mtx)
    
    return np.array(error_mtx)



def MMWs_BBOXs_match(MMWs, BBOXs, img):
    # cal the error matrix between bboxs and mmws
    error_matrix = CalErrorMatrix(MMWs, BBOXs) # size: m x n

    if error_matrix.size == 0: # one of them(MMWs, BBOXs) is empty
            matches_idx_list, u_MMWs_idx_list, u_BBOXs_idx_list = np.array([]), \
                                       np.arange(len(MMWs), dtype=int), \
                                       np.arange(len(BBOXs), dtype=int),
    else: # assign MMW to BBOX, one to one 
        matches_idx_list, u_MMWs_idx_list, u_BBOXs_idx_list = linear_assignment(error_matrix) # matchs, unmatch_MMWs, unmatch_BBOXs
    
    """ for matches """
    for m_idx, b_idx in matches_idx_list:
        MMWs[m_idx].BBOX_ID = BBOXs[b_idx].ID
        MMWs[m_idx].matched_BBOX = BBOXs[b_idx]

        BBOXs[b_idx].MMW_ID = MMWs[m_idx].ID
        BBOXs[b_idx].matched_MMW = MMWs[m_idx]
    
    # """ for unmatch MMWs """
    # u_MMWs = [] # save all unmatch MMW(); [MMW(), ...]
    # for idx in u_MMWs_idx_list:
    #     u_MMWs.append(MMWs[idx])

    # """ for unmatch MMWs """
    # u_BBOXs = [] # save all unmatch BBOX(); [BBOX(), ...]
    # for idx in u_BBOXs_idx_list:
    #     u_BBOXs.append(BBOXs[idx])
    
    return MMWs, BBOXs, u_MMWs_idx_list, u_BBOXs_idx_list, matches_idx_list


