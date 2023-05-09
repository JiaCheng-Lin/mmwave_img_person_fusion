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

sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_POSENET_RELEASE/', 'main'))
sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_POSENET_RELEASE/', 'data'))
sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_POSENET_RELEASE/', 'common'))
sys.path.insert(0, osp.join(r'C:\TOBY\jorjin\3DMPPE_POSENET_RELEASE/common', 'utils'))
from config1 import cfg
import model1
import dataset1 
from pose_utils import process_bbox, pixel2cam
from vis import vis_keypoints, vis_3d_multiple_skeleton

import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

cudnn.benchmark = True

def PoseNetInit():
    # MuCo joint set
    joint_num = 21
    model_path = r"C:\TOBY\jorjin\3DMPPE_POSENET_RELEASE\model/snapshot_24.pth.tar"
    model = model1.get_pose_net(cfg, False, joint_num)
    model = DataParallel(model).cuda()
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['network'])
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])

    return model, transform


def getSkeletonFromPoseNet(model, online_im, online_tlwhs, root_depth_list, \
                        transform, original_img_height=480, original_img_width=640, \
                        focal=[674.97055585, 674.97055585]): # focal: cal by zhang-calibatrion
    princpt = [original_img_width/2, original_img_height/2] # [304.6173947, 248.93570592] 

    output_pose_2d_list = []
    output_pose_3d_list = []
    for i, tlwh in enumerate(online_tlwhs):
        bbox = process_bbox(np.array(tlwh), original_img_width, original_img_height)
        img, img2bb_trans = dataset1.generate_patch_image(online_im, bbox, False, 1.0, 0.0, False) # poseNet
        img = transform(img).cuda()[None,:,:,:]

        """ poseNet """
        # forward
        with torch.no_grad():
            pose_3d = model(img) # x,y: pixel, z: root-relative depth (mm)

        # inverse affine transform (restore the crop and resize)
        pose_3d = pose_3d[0].cpu().numpy()
        pose_3d[:,0] = pose_3d[:,0] / cfg.output_shape[1] * cfg.input_shape[1]
        pose_3d[:,1] = pose_3d[:,1] / cfg.output_shape[0] * cfg.input_shape[0]
        pose_3d_xy1 = np.concatenate((pose_3d[:,:2], np.ones_like(pose_3d[:,:1])),1)
        img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0,0,1]).reshape(1,3)))
        pose_3d[:,:2] = np.dot(np.linalg.inv(img2bb_trans_001), pose_3d_xy1.transpose(1,0)).transpose(1,0)[:,:2]
        output_pose_2d_list.append(pose_3d[:,:2].copy())
        
        # root-relative discretized depth -> absolute continuous depth
        pose_3d[:,2] = (pose_3d[:,2] / cfg.depth_dim * 2 - 1) * (cfg.bbox_3d_shape[0]/2) + root_depth_list[i]
        pose_3d = pixel2cam(pose_3d, focal, princpt)
        output_pose_3d_list.append(pose_3d.copy())

    # visualize 2d poses
    joint_num = 21
    skeleton = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )
    vis_img = online_im.copy()
    for n in range(len(online_tlwhs)):
        vis_kps = np.zeros((3, joint_num))
        vis_kps[0,:] = output_pose_2d_list[n][:,0]
        vis_kps[1,:] = output_pose_2d_list[n][:,1]
        vis_kps[2,:] = 1
        vis_img = vis_keypoints(vis_img, vis_kps, skeleton)
    # cv2.imwrite('output_pose_2d.jpg', vis_img)
    cv2.imshow("vis_img", vis_img)
    
    # visualize 3d poses
    vis_kps = np.array(output_pose_3d_list)
    np.save("./utils/vis_kps.npy", vis_kps)
    # print(vis_kps)
    # vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')
    # fig = vis_3d_multiple_skeleton_realTime(fig, ax, vis_kps, np.ones_like(vis_kps), skeleton, 'output_pose_3d (x,y,z: camera-centered. mm.)')

    

    return online_im



def vis_3d_multiple_skeleton_realTime(fig, ax, kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, c=colors[l], linewidth=2)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], c=colors[l], marker='o')
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], c=colors[l], marker='o')

            # plt.pause(0.0001) #Note this correction
    return fig
    # if filename is None:
    #     ax.set_title('3D vis')
    # else:
    #     ax.set_title(filename)

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Z Label')
    # ax.set_zlabel('Y Label')
    # ax.legend()

    # plt.show()
    
    # cv2.waitKey(0)