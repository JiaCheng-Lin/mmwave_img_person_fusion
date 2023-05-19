import json 
import os
import cv2

abs_path = r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\img_mmwave_data/"
    

def saveData(img, mmwave, save_path, idx):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_name = os.path.join(save_path, 'image_{:08d}.jpg')
    mmwave_name = os.path.join(save_path, 'mmwave_{:08d}.json')

    cv2.imwrite(img_name.format(idx), img)
    with open(mmwave_name.format(idx), 'w') as f:
        json.dump(mmwave, f)



    
