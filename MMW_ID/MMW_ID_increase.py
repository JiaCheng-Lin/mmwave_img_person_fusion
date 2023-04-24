import numpy as np 
import cv2
import time

import numpy as np
import os
import json
import copy

import sys
sys.path.append("../utils")
from MMW import MMW
from mmwave_pts_visualization import *


folderName = "20230316_013617"#""  20230316_021729 20230418_145353
abs_path = r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\img_mmwave_data/"
data_dir = abs_path+'{}'.format(folderName)
bg = cv2.imread(r"C:\TOBY\jorjin\MMWave\mmwave_webcam_fusion\inference\byteTrack_mmwave\inference\utils/mmwave_bg_1.png")

def main():
	# data = np.array(np.load('Detections.npy'))[:,10:300,:]
	record = RecordRadarPoints(4)
	for filename in os.listdir(data_dir):
		if filename.startswith("image_"):
			img_path = os.path.join(data_dir, filename)
			img = cv2.imread(img_path)

			mmwave_filename = f"mmwave{filename[5:-4]}.json" # get mmwave name 
			mmwave_path = os.path.join(data_dir, mmwave_filename)
			with open(mmwave_path, 'r') as f: ## get mmwave data
				sync_mmwave_json = json.load(f)
			bg_original = copy.deepcopy(bg)
			bg_new1 = copy.deepcopy(bg)

			## show original radar data
			MMW_pts = process_MMW_json(sync_mmwave_json) # MMW points per frame
			for pt in MMW_pts:
				bg_original = pt.draw(bg_original)
			cv2.imshow("original", bg_original)
			
			# record points
			MMW_pts = record.update(MMW_pts)
			for pt in MMW_pts:
				# print(pt.ID)
				bg_new1 = pt.draw(bg_new1)
			cv2.imshow("bg_new1", bg_new1)

			cv2.imshow("img", img) 
			if cv2.waitKey(60) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

# If an ID disappears for "X" consecutive frames 
# mark it and it will not appear again / freeze ID
class RecordRadarPoints:
	def __init__(self, set_frame):
		self.frame_limit = set_frame # disappear "X frame" will be delete
		self.max_ID = 0 
		self.freeze_ID = [] # ID mark list, freeze ID
		self.appear_ID = [] # save the ID have appeared
		self.disappear_ID_cnt = np.zeros(100, dtype=int)
		self.map_ID = np.zeros(100, dtype=int)
		for i in range(100):
			self.map_ID[i] = i
	
	def update(self, pts_frame): # pts_frame: MMW class
		appear_this_frame = []
		MMW_pts = copy.deepcopy(pts_frame)
		for pt in pts_frame:
			appear_this_frame.append(self.map_ID[pt.ID])
			if self.map_ID[pt.ID] not in self.appear_ID:
				self.appear_ID.append(self.map_ID[pt.ID])
			else:
				self.disappear_ID_cnt[self.map_ID[pt.ID]] = 0
			self.max_ID = max(self.max_ID, self.map_ID[pt.ID])

		for ID in self.appear_ID:
			if ID not in appear_this_frame:
				self.disappear_ID_cnt[ID] += 1
				if self.disappear_ID_cnt[ID] == self.frame_limit:
					self.freeze_ID.append(ID)

		self.freeze_ID = list(set(self.freeze_ID))
		for i in range(len(pts_frame)):
			if self.map_ID[pts_frame[i].ID] in self.freeze_ID:
				self.map_ID[pts_frame[i].ID] = self.max_ID+1
				MMW_pts[i].ID = self.map_ID[pts_frame[i].ID]
				self.max_ID += 1
			MMW_pts[i].ID = self.map_ID[pts_frame[i].ID]
		
		## TODO: 被佔用的(map對應到的)先不用改，等到相同的ID出現後(出現重複),再去increase

		for pt in pts_frame:
			# print(pt.ID)
			# print(self.map_ID[pt.ID]!= pt.ID)
			# print(self.map_ID[self.map_ID[pt.ID]] == self.map_ID[pt.ID])
			if self.map_ID[pt.ID]!= pt.ID and self.map_ID[self.map_ID[pt.ID]] == self.map_ID[pt.ID]:
				self.map_ID[self.map_ID[pt.ID]] = self.max_ID+1
				# print(self.max_ID)
				# print(self.map_ID[self.map_ID[pt.ID]])
				self.max_ID += 1
		# print(self.map_ID)
		
		return MMW_pts



                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

if __name__ == '__main__':
	main()