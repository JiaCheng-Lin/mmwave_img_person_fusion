import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_ROOTNET_RELEASE/', 'main'))
sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_ROOTNET_RELEASE/', 'data'))
sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_ROOTNET_RELEASE/', 'common'))
sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_ROOTNET_RELEASE/common', 'utils'))
from model import get_pose_net 
from config import cfg
from pose_utils import process_bbox
from dataset import generate_patch_image

cudnn.benchmark = True

def RootNetInit():
    joint_num = 21
    model_path = r"C:\TOBY\jorjin\3DMPPE_ROOTNET_RELEASE\model/snapshot_18.pth.tar"
    model = get_pose_net(cfg, False)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    model.eval()

    # for image processing
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])

    return model, transform

def createimage(w,h):
	size = (w, h, 1)
	img = np.ones((w,h,3),np.uint8)*255
	return img
    

def getDepthFromRootNet(model, online_im, online_tlwhs, transform, \
                        original_img_height=480, original_img_width=640, \
                        focal=[674.97055585, 674.97055585]): # focal: cal by zhang-calibatrion
    princpt = [original_img_width/2, original_img_height/2] #  [304.6173947, 248.93570592] # 

    root_depth_list = []
    frame = createimage(600, original_img_width)
    for i, tlwh in enumerate(online_tlwhs):
        bbox = process_bbox(np.array(tlwh), original_img_width, original_img_height)
        img, img2bb_trans = generate_patch_image(online_im, bbox, False, 0.0)  # rootNet
        img = transform(img).cuda()[None,:,:,:]
        k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
        k_value = torch.FloatTensor([k_value]).cuda()[None,:]

        """ rootNet """
        # forward
        with torch.no_grad():
            root_3d = model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
        img = img[0].cpu().numpy()
        root_3d = root_3d[0].cpu().numpy()
        root_depth_list.append(root_3d[2])

        print(int(root_3d[0]), int(root_3d[1]), int(root_3d[2]/10))
        cx, cy = tlwh[0]+tlwh[3]//2, tlwh[1]+tlwh[2]//2
        print("cx, cy", cx, cy)
        cv2.circle(frame, (int(cx), 600-int(root_3d[2]/10)), 5, (0, 255, 0), -1) 
        cv2.imshow("yz", frame)

    


        # save output in 2D space (x,y: pixel)
        # vis_img = img.copy()
        # vis_img = vis_img * np.array(cfg.pixel_std).reshape(3,1,1) + np.array(cfg.pixel_mean).reshape(3,1,1)
        # vis_img = vis_img.astype(np.uint8)
        # vis_img = vis_img[::-1, :, :]
        # vis_img = np.transpose(vis_img,(1,2,0)).copy()
        # vis_root = np.zeros((2))
        # vis_root[0] = root_3d[0] / cfg.output_shape[1] * cfg.input_shape[1]
        # vis_root[1] = root_3d[1] / cfg.output_shape[0] * cfg.input_shape[0]
        
        # print(vis_root[0], vis_root[1])
        # cv2.circle(vis_img, (int(vis_root[0]), int(vis_root[1])), radius=5, color=(0,255,0), thickness=-1, lineType=cv2.LINE_AA)
        cv2.putText(online_im, str(round(root_3d[2]/1000,4)), (int(tlwh[0]), int(tlwh[1]+20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 187), 1, cv2.LINE_AA)

        # cv2.imshow("output_root_2d_" + str(i), vis_img)
        # cv2.imwrite('output_root_2d_' + str(n) + '.jpg', vis_img)
        # print('Root joint depth: ' + str(root_3d[2]/1000) + ' m')
    print("======")

    return online_im, root_depth_list
