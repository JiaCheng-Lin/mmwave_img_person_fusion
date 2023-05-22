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
                vec_dot = vx*img_vx+vy*img_vy # angle < 90 degree
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

# direction check: if two dir degree>90(dot<0) -> unmatch
# cal the degree/dot of two dir
def sameDir(MMW_dir, BBOX_dir):
    if MMW_dir and BBOX_dir: # if two dir exists
        dir_dot = np.dot(np.array(MMW_dir), np.array(BBOX_dir))
        #  print("dot", dir_dot)  # print("degree", np.arccos(dir_dot/np.linalg.norm(np.array(mmw_cls.Dir))/np.linalg.norm(np.array(bbox_cls.Dir)))*180/np.pi)
        if dir_dot < 0: # degree > 90
            return False 
    return True # have not dir / dir_dot>=0 (degree<=90)

def CalErrorMatrix(MMWs, BBOXs, error_threshold=150):
    unmatch_value = 1e6 
    error_mtx = [] # m x n
    
    for mmw_cls in MMWs:
        row_error_mtx = [] # save the ERRORs: one MMW() to each BBOX() 
        for bbox_cls in BBOXs:
            if checkWithinBbox(mmw_cls, bbox_cls) == True: # MMW() in bbox
                pixel_error, meter_error = cal_BBOX_MMW_error(mmw_cls, bbox_cls) # cal th err
                error = pixel_error*1 + meter_error*100 # weight

                if error>error_threshold: # or sameDir(mmw_cls.Dir, bbox_cls.Dir)==False: 
                    row_error_mtx.append(unmatch_value) #  give a large number to <error mtx>
                else:
                    row_error_mtx.append(error) # add to error mtx

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
        # MMWs[m_idx].matched_BBOX = BBOXs[b_idx]

        BBOXs[b_idx].MMW_ID = MMWs[m_idx].ID
        # BBOXs[b_idx].matched_MMW = MMWs[m_idx]
    
    """ for unmatch MMWs """
    u_MMWs = [] # save all unmatch MMW(); [MMW(), ...]
    for idx in u_MMWs_idx_list:
        u_MMWs.append(MMWs[idx])

    """ for unmatch MMWs """
    u_BBOXs = [] # save all unmatch BBOX(); [BBOX(), ...]
    for idx in u_BBOXs_idx_list:
        u_BBOXs.append(BBOXs[idx])
    
    return MMWs, BBOXs, u_MMWs, u_BBOXs, matches_idx_list



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
        
        cv2.putText(im0, "-"+str(ID_mmwave)+" ", (l+20, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 0), 1, cv2.LINE_AA)
        # cv2.putText(im0, str(real_dis), (l+50, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 255, 0), 1, cv2.LINE_AA)

        new_mmwave_pts_list.append([px, py, str(ID_img)+"_"+str(ID_mmwave), (232, 229, 26)]) # give green color for vis to distinguish
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
                new_mmwave_pts_list.append([px, py, str(ID_img)+"_"+str(ID_mmwave), (0, 143, 255)]) # add to mmwave visualization
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

