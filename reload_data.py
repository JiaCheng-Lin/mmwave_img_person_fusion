import argparse
import os
import os.path as osp
import time
import cv2
import threading
import torch

from loguru import logger

# path: c:\toby\jorjin\object_tracking\bytetrack\yolox\__init__.py
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer

"""mmwave""" 
from socket import *
from datetime import datetime
import json
# import time
import numpy as np
from utils.mmwave import * # import mmwave utils (functions)
from utils.mmwave_pts_visualization import *
import copy
"""mmwave""" 

## for regression model prediction. (mmwave pt project to image)
from joblib import dump, load # save and load model.
from sklearn.model_selection import train_test_split # split data to train&test
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures # Polynomial
from sklearn.pipeline import make_pipeline

## save data
from utils.save_data import *

# MMW vis
sys.path.append(r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\inference\utils")
from MMW import MMW, json2MMWCls

## BBOX
from BBOX import BBOX, list2BBOXCls

from matching import MMWs_BBOXs_match, cal_BBOX_MMW_error


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="./videos/video_11_46_24.avi", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    ### DEFAULT: "bytetrack_x_mot17" 
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=r'C:\TOBY\jorjin\object_tracking\ByteTrack\exps\example\mot/yolox_x_mix_det.py', # yolox_tiny_mix_det
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "-c", 
        "--ckpt", 
        default=r'C:\TOBY\jorjin\object_tracking\ByteTrack\pretrained/bytetrack_x_mot17.pth.tar',  # bytetrack_tiny_mot17
        type=str, 
        help="ckpt for eval"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=10, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    # # about fps issues: https://github.com/ifzhang/ByteTrack/issues/156
    # # the timer calculation is wrong.
    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        # s = datetime.now()  
        # # spend about 0.06 s
        timer.tic()
        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        # e = datetime.now() 
        # print("Transform time", (e-s).total_seconds())
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        # spend about 0.048 s
        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, timestamp)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

def distance_finder(focal_length, real_object_width, width_in_frame): # fake distance finder
    distance = (real_object_width * focal_length) / width_in_frame
    return distance

def get_center_pt_list(online_im, online_ids, online_tlwhs):
    center_pt_list = [] # [[centerPt_u, centerPt_v, fake_dis, ID_img], ]
    for idx, tlwh in enumerate(online_tlwhs):
        center_pt_x, center_pt_y = int(tlwh[0]+tlwh[2]/2), int(tlwh[1]+tlwh[3]/2)
        bottom_pt = int(tlwh[1]+tlwh[3])-20
        # cv2.circle(online_im, (center_pt_x, bottom_pt), 2, (0, 0, 255), 5)
        cv2.circle(online_im, (center_pt_x, center_pt_y), 2, (0, 0, 255), 5)

        """  estimate fake distance using bbox  """ # reference: (github) Yolov4-Detector-and-Distance-Estimator
        focal_person = 1035 # pixels, calculation see main_BT.py file
        PERSON_WIDTH = 16  # inches
        fake_dis =  distance_finder(focal_person, PERSON_WIDTH, int(tlwh[2]))
        fake_dis = round(fake_dis*0.0254, 2) # inch -> meter
        # cv2.putText(online_im, "cam "+str(fake_dis), (center_pt_x, center_pt_y-50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
        # add to list
        # center_pt_list.append([center_pt_x, bottom_pt, fake_dis, online_ids[idx], tlwh])
        center_pt_list.append([center_pt_x, center_pt_y, fake_dis, online_ids[idx], tlwh])

    return center_pt_list, online_im

### Need to improve code 
def draw_frame_arrow(pre, cur, img):
    # pre, cur: [center_pt_x, center_pt_y, fake_dis, online_ids[idx], tlwh]
    new_list = copy.deepcopy(cur)
    for i, (x1, y1, _, id1, _) in enumerate(cur):
        new_list[i].append((0, 0)) # init, no direction.
        if pre:
            for x2, y2, _, id2, _ in pre:
                if id1 == id2:
                    # draw direction arrow
                    Vx, Vy = x1-x2, y1-y2
                    dis = math.sqrt(Vx**2+Vy**2)
                    # print("dis: ", math.sqrt(Vx**2+Vy**2)) # person stand -> within about 10 pixel 
                    if Vx == 0 and Vy == 0 or dis <= 5: # if dis<5 pixel, skip it
                        break

                    Vx, Vy = Vx/math.sqrt((Vx**2+Vy**2))*20, Vy/(math.sqrt(Vx**2+Vy**2))*20  # normalize the length in fig
                    end_point = (x1+int(Vx), y1+int(Vy)) # the direction
                    cv2.arrowedLine(img, (x1, y1), end_point, (0, 255, 0), 3) 
                    new_list[i][-1] = end_point # new direction to the center_list
                    break

    # print("new_list", new_list)
    return new_list

# define a function for horizontally concatenating images of different heights 
def hconcat_resize(img_list, interpolation = cv2.INTER_CUBIC):
    # take minimum hights
    h_min = min(img.shape[0] 
                for img in img_list)
    
    # image resizing 
    im_list_resize = [cv2.resize(img,
                    (int(img.shape[1] * h_min / img.shape[0]), h_min),
                    interpolation = interpolation) 
                    for img in img_list]
    
    # return final image
    return cv2.hconcat(im_list_resize)

## process the video or webcam flow
def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    if args.save_result:
        os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    if args.demo == "webcam": 
        origin_vid_writer = cv2.VideoWriter(
            osp.join(save_folder, "origin_camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    # mmwave_json = pre_mmwave_json = None # initialize mmwave_json data
    pre_center_pt_list = []
    center_pt_list = []
    previous_ID_matches  = [] # record previous ID matches
    pre_BBOXs, BBOXs = [], [] # record previous/current BBOXs infomation
    ID_matches  = [] # record ID matches

    # # read mmwave background image
    bg = cv2.imread(r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\inference\utils/mmwave_bg_1.png")
    
    # # regression model initialization (mmwave pts project to img)
    regressor = load(r'C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\cal_tranform_matrix\data/data_2023_03_09_13_05_11.joblib') 
    t_total = 0 # time_error_sum
    t_cnt = 0 # time count

    prev_frame_time = 0
    new_frame_time = 0

    ## save path for image & mmwave
    folderName = "20230528_195907_you"  
    abs_path = r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\img_mmwave_data/"
    data_dir = abs_path+'{}'.format(folderName)

    ## count the number of unmatch bbox(person)
    non_match_bbox = 0
    total_bbox = 0

    # count the number of ID Switch
    id_switch = 0

    ## save bbox video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('../img_mmwave_data/'+folderName+'/output.avi', fourcc, 10.0, (1960,  480))
    
    out1 = cv2.VideoWriter('../img_mmwave_data/'+folderName+'/output_clear.avi', fourcc, 10.0, (1960,  480))

    out_pos = cv2.VideoWriter('../img_mmwave_data/'+folderName+'/output_pos.avi', fourcc, 10.0, (1300,  480))

    out_centroid = cv2.VideoWriter('../img_mmwave_data/'+folderName+'/output_centroid.avi', fourcc, 10.0, (1300,  480))

    out_original = cv2.VideoWriter('../img_mmwave_data/'+folderName+'/output_original.avi', fourcc, 10.0, (1300,  480))

    text_record_path = '../img_mmwave_data/'+folderName+'/output.txt'
    # f = open(text_record_path, 'w+')


    # load regression model
    import utils.load_regression_model as LRM
    ### Camera to Radar
    bbox2MMW_model_name =  'model_bbox2mmw_0.191.ckpt'  # <- 20230528    # 'model_bbox2mmw_0.230.ckpt' # <- 20230527  # 'model_bbox2mmw_0.204.ckpt' # <- 20230526
    bbox2MMW_model = LRM.get_bbox2MMW_regression_model(bbox2MMW_model_name, input_dim=2)

    ### Radar to Camera
    MMW2bbox_model_name =  'model_mmw2bbox_26.917.ckpt' # <- 20230528   # 'model_mmw2bbox_23.936.ckpt' # 'model_mmw2bbox_25.740.ckpt' 
    MMW2bbox_model = LRM.get_MMW2bbox_regression_model(MMW2bbox_model_name, input_dim=6)
    
    match_cnt = 0
    pixel_error_sum, meter_error_sum = 0, 0

    ## UID assignment
    from UID import UID_assignment
    UID_number = 0 # init
    BBOXs_UID = {}
    MMWs_UID = {}
    

    for filename in os.listdir(data_dir):

        if filename.startswith("image_"):

            img_path = os.path.join(data_dir, filename)
            frame = cv2.imread(img_path)

            mmwave_filename = f"mmwave{filename[5:-4]}.json" # get mmwave name 
            mmwave_path = os.path.join(data_dir, mmwave_filename)

            ## get mmwave data
            with open(mmwave_path, 'r') as f:
                sync_mmwave_json = json.load(f)

            # # image process: get bbox and id of each person; person detection and tracking, about 0.1s per frame
            outputs, img_info = predictor.inference(frame, timer)

            ## mmwave pts visualization by OpenCV, time consume: about 1ms. 
            bg_copy = copy.deepcopy(bg)
            # mmwave_pt_visual = draw_mmwave_pts(bg_copy, ori_mmwave_json)
            # mmwave_pt_visual = draw_mmwave_pts_sync(bg_copy, sync_mmwave_json)
            # cv2.imshow("mmwave", mmwave_pt_visual)

            if outputs[0] is not None: # bbox exists
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                
                online_im = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
                
                # cv2.imshow("byteTrack", online_im)
                # 
                # if center_pt_list:
                #     pre_center_pt_list = center_pt_list

                # center_pt_list, online_im = get_center_pt_list(online_im, online_ids, online_tlwhs)

                # # Draw the direction of each person's center point between frames. 
                # new_center_pt_list = draw_frame_arrow(pre_center_pt_list, center_pt_list, online_im)
                # # print("new center pt", new_center_pt_list) # center_pt_x, center_pt_y, fake_dis, online_ids[idx], tlwh, (dir_x, dir_y)

                """ DO Transform! (project bbox pts to MMW) """ # time consuming: < 10ms
                s = datetime.now() 
                pre_BBOXs = BBOXs # Record previous BBOXs, you can get the directon / still(?)
                BBOXs = list2BBOXCls(online_ids, online_tlwhs, pre_BBOXs) # convert original data to list[BBOX(), BBOX(), ...]
                BBOXs = LRM.predict_pos(bbox2MMW_model, BBOXs) # ## predict Xr, Yr in Radar for each BBOX() 

                r = 80
                # cv2.rectangle(online_im, (r, r), (640-r, 480-r), color=(100,0,0), thickness=2)

                # visualization
                for bbox_cls in BBOXs: 
                    bg_copy = bbox_cls.drawInRadar(bg_copy, pt_color=(0, 143, 255), pt_size=3, showArrow=False) # draw estimated Radar points(Xr, Yr) in radar plane
                    online_im = bbox_cls.drawInCamera(online_im, pt_color=(0, 0, 255), pt_size=5, showArrow=False) # draw bbox centroid in image 
                e = datetime.now()  
                cv2.imshow("online_im", online_im)

                online_im_cp = copy.deepcopy(online_im)
                # print("bbox predict time", (e-s).total_seconds())

            else:
                timer.toc()
                online_im = img_info['raw_img']


            """ DO Transform! (project mmwave pts to img) """ # y_list: [[px, py, real_dis, ID], ], 
            # # start = datetime.now()
            # # online_im, _ = process_mmwave(ori_mmwave_json, online_im, origin_px=6.0, origin_py=1.0, regressor=regressor)
            # online_im, estimated_uv_list = process_mmwave_sync(sync_mmwave_json, online_im, origin_px=6.0, origin_py=1.0, regressor=regressor)
            # online_im, _= process_mmwave_regression(sync_mmwave_json, online_im, origin_px=6.0, origin_py=1.0, regression_model=regression_model)
            # # end = datetime.now()  
            # # print("Transform time", (end-start).total_seconds()) # about 1ms.

            """ Match, cal error: find the corresponding person """
            ### px, py, real_dis, ID_mmwave <=> center_u, center_v, esti_dis, ID_img, tlwh, dir_pt (xy_list <=> center_pt_list)



            """ NEW Version: DO Transform (project mmw pts to image) """  # time consuming: < 10ms
            s1 = datetime.now()   #
            MMWs = json2MMWCls(sync_mmwave_json) # convert json data to list[MMW(), MMW(), ...]
            MMWs = LRM.predict_pixel(MMW2bbox_model, MMWs)  ## predict Xc, Yc in image for each MMW() 
            
            for mmw_cls in MMWs: # visualization
                bg_copy = mmw_cls.drawInRadar(bg_img=bg_copy, pt_color=(255, 0, 0), pt_size=3, showArrow=False)  ## draw time_sync MMW points
                online_im = mmw_cls.drawInCamera(img=online_im, pt_color=(204, 0, 204), pt_size=2, text="", showArrow=False) # draw estimated points(Xc,Yc) in image
            # cv2.imshow("online_im", online_im)
            cv2.imshow("bg_copy", bg_copy)

            e1 = datetime.now()  
            # print("predict time", (e1-s1).total_seconds())


            img_white = 255*np.ones((480, 20, 3), np.uint8)
            # function calling
            im_pos = hconcat_resize([bg_copy, img_white, online_im_cp])
            cv2.imshow("img_pos", im_pos)
            # out_pos.write(im_pos)



            if outputs[0] is not None:
                """ NEW Version Match method """ 
                # input: RADAR: [MMW(), MMW(), ...]
                #          CAM: [BBOX(), BBOX(), ...]

                s2 = datetime.now()   #
                MMWs, BBOXs, u_MMWs_idx_list, u_BBOXs_idx_list, matches_idx_list = MMWs_BBOXs_match(MMWs, BBOXs, online_im)
                
                ## vis
                for idx, bbox_cls in enumerate(BBOXs):  # draw in cam image
                    online_im = bbox_cls.drawCorrespondingMID(online_im) # draw matched MMW_ID in image

                bg_match = copy.deepcopy(bg) ## mmw plane image for MATCH
                for i, bbox_cls in enumerate(BBOXs): # unmatch BBOX -> draw estimated (Xr, Yr) in Radar image
                    if i in u_BBOXs_idx_list:
                        bg_match = bbox_cls.drawInRadar(bg_match) # draw unmatch_BBOX estimated (Xr, Yr) in radar plane image.
                for mmw_cls in MMWs: 
                    bg_match = mmw_cls.drawCorrespondingCID(bg_match) # draw matched BBOX_ID in radar plane image
                cv2.imshow("bg_match", bg_match)
                
                e2 = datetime.now()  
                # print("predict time", (e2-s2).total_seconds())

                """ cal error """ # for Quantitative results
                for bbox_cls in BBOXs:
                    if bbox_cls.matched_MMW and bbox_cls:
                        pixel_error, meter_error = cal_BBOX_MMW_error(bbox_cls.matched_MMW, bbox_cls)
                        pixel_error_sum += pixel_error
                        meter_error_sum += meter_error
                        match_cnt += 1
                        # print(pixel_error, meter_error)
                        print("pixel_mean_error", pixel_error_sum/match_cnt)
                        print("meter_mean_error", meter_error_sum/match_cnt)
                        # with open(text_record_path, 'w+') as f:
                        #     f.write(str(match_cnt) + ", pixel_mean_error: " + str(pixel_error_sum/match_cnt) )
                        #     f.write(str(match_cnt) + ", meter_mean_error: " + str(meter_error_sum/match_cnt) )


                """ UID """
                BBOXs, MMWs, BBOXs_UID, MMWs_UID, UID_number = UID_assignment(MMWs, BBOXs, matches_idx_list, BBOXs_UID, MMWs_UID, UID_number)
                print("MMWs_UID", MMWs_UID)
                print("BBOXs_UID", BBOXs_UID)

                # vis
                bg_UID = copy.deepcopy(bg)
                for i, bbox_cls in enumerate(BBOXs): # unmatch BBOX -> draw estimated (Xr, Yr) in Radar image
                    # if i in u_BBOXs_idx_list:
                    bg_UID = bbox_cls.drawUIDInRadar(bg_UID) # draw unmatch_BBOX estimated (Xr, Yr) in radar plane image.
                for i, mmw_cls in enumerate(MMWs): 
                    if i in u_MMWs_idx_list:
                        bg_UID = mmw_cls.drawUID(bg_UID) # draw matched BBOX_ID in radar plane image
                # cv2.imshow("bg_UID", bg_UID)



                """ vis """

                 # original MMW img
                original_MMW_bg = copy.deepcopy(bg)
                for mmw_cls in MMWs: # visualization
                    original_MMW_bg = mmw_cls.drawInRadar(bg_img=original_MMW_bg, pt_color=(200, 0, 0), pt_size=4, showArrow=False)  ## draw time_sync MMW points

                
                # im_original = hconcat_resize([original_MMW_bg, img_white, frame])
                # out_original.write(im_original)

                # im_centroid = hconcat_resize([original_MMW_bg, img_white, online_im])
                # cv2.imshow("im_centroid", im_centroid)
                # out_centroid.write(im_centroid)


                img_white = 255*np.ones((480, 20, 3), np.uint8)
                # function calling
                im_debug = hconcat_resize([original_MMW_bg, img_white, bg_UID, img_white, online_im])

                # show the Output image
                cv2.imshow('im_debug', im_debug)
                out.write(im_debug) # save UID_result


                """ FOR USER VIEW img """
                user_im = copy.deepcopy(frame)
                for bbox in BBOXs:
                    user_im = bbox.drawBBOXInCamera(user_im)


                User_res_im = hconcat_resize([original_MMW_bg, img_white, bg_UID, img_white, user_im])  
                cv2.imshow('User_res_im', User_res_im)
                # out1.write(User_res_im)


                # # new_mmwave_pts_list: show mmwave pts, ID_matches: record ID_match, previ
                # online_im, new_mmwave_pts_list, previous_ID_matches, non_match_bbox, total_bbox = pt_match_valid(estimated_uv_list, new_center_pt_list, online_im, non_match_bbox, total_bbox, previous_ID_matches)

                # ### after match, show a new mmwave_pt_visual
                # new_bg_copy = copy.deepcopy(bg)
                # # new_mmwave_pt_visual = draw_mmwave_pts_sync(new_bg_copy, xy_list=new_mmwave_pts_list)
                # cv2.imshow("new_mmwave_pt_visual", new_mmwave_pt_visual)
            """!!! mmwave process !!!"""

            # if args.save_result:
            #     vid_writer.write(online_im)
            #     if args.demo == "webcam": 
            #         origin_vid_writer.write(frame)
            # cv2.imshow("online_im", online_im)
            ch = cv2.waitKey(1)
            # if ch == ord("a") or ch == ord("A"):
            #     non_match_bbox+=1
            
            # out.write(online_im) # save video after obj detection

            # ## count the number of id switch
            # if ch >= 49 and ch <= 57:
            #     id_switch += int(ch)-48
            #     print("id_switch", id_switch)

            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                # print("non_match", non_match_bbox)
                # print("total", total_bbox)
                out.release()
                out1.release()
                f.close()
                break

        else:
            break
        frame_id += 1

    out.release()
    out1.release()
    f.close()
    # print("non_match", non_match_bbox)
    # print("total", total_bbox)
        

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")

## receive mmwave data  
def receive_data(udp_socket, BUFSIZE):
    while True:
        data, addr = udp_socket.recvfrom(BUFSIZE)
        
        process_data(data) 

def process_data(data):
    global mmwave_json, pre_mmwave_json # must be declared as global var, otherwise it is a local variable in function
    
    if data:
        if mmwave_json != "":
            pre_mmwave_json = mmwave_json
            
        mmwave_json = json.loads(data)

        # # cal the interval of mmwave.
        # if pre_mmwave_json and mmwave_json:
        #     pre_t = datetime.strptime(pre_mmwave_json['TimeStamp'], "%H:%M:%S:%f").time()
        #     now_t = datetime.strptime(mmwave_json['TimeStamp'], "%H:%M:%S:%f").time()

        #     t = cal_time_error(now_t, pre_t)
        #     if t>0.07:
        #         print(t, now_t, pre_t)
        # print("mmwave:", mmwave_json["TimeStamp"])

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join("../"+exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    if args.save_result:
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")

        # if "head.reid_classifier.weight" in ckpt["model"]:  # TODO: remove checkpoint of ReID classifier
        #     ckpt["model"].pop("head.reid_classifier.weight")
        # if "head.reid_classifier.bias" in ckpt["model"]:  # TODO: remove checkpoint of ReID classifier
        #     ckpt["model"].pop("head.reid_classifier.bias")
        # model.load_state_dict(ckpt["model"], strict=False)  # TODO: set strict=False for missing keys of classifier

        # load the model state dict
        model.load_state_dict(ckpt["model"]) # original 
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        # trt_file = osp.join(output_dir, "model_trt.pth")
        trt_file_name = str(args.exp_file.split('/')[-1][:-3])
        trt_file = osp.join(r"C:\TOBY\jorjin\object_tracking\ByteTrack\YOLOX_outputs", trt_file_name, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None
    
    # """ receive mmwave data """
    # udp_socket = socket(AF_INET, SOCK_DGRAM)
    # udp_socket.bind(("0.0.0.0", 6000)) # ip and port of jorjin mmwave app
    # BUFSIZE = 2048
    # # # Add a thread for receiving data from mmwave radar
    # threading.Thread(target=receive_data, args=(udp_socket, BUFSIZE), daemon=True).start()
    # """ receive mmwave data """

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    # mmwave parameters init
    mmwave_json = pre_mmwave_json = ""

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)