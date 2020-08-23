import numpy as np 
import random, sys
sys.path.append('../context_profile')
from get_data_per_cat import compare_iou
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy



def generate_appear_box(im_info, gt_boxes, cnt=1):
	H, W = im_info[0], im_info[1]
	return_boxes = []
	iou_list = []
	for _ in range(cnt):
		h = random.randint(int(0.1*H), int(0.9*H))
		w = random.randint(int(0.1*W), int(0.9*W))
		y1 = random.randint(int(0.01*H), int(0.99*H-h))
		x1 = random.randint(int(0.01*W), int(0.99*W-w))
		x2, y2 = x1+w, y1+h 
		new_box = [x1,y1,x2,y2,0]
		return_boxes.append(new_box)
		ious = compare_iou(gt_boxes, new_box)
		iou_list.append(np.max(ious))
	return return_boxes, iou_list

if __name__ == '__main__':
	im_info = [200,400,1.2]
	gt_boxes = np.array([[100,100, 350,150, 2]])
	H, W = im_info[0], im_info[1]
	im = np.zeros([H,W,3])
	x1, y1, x2, y2 = gt_boxes[0][:4]
	im[y1:y2,x1:x2]=1
	plt.subplot(2,1,1)
	plt.axis('off')
	plt.imshow(im)
	plt.title('original')

	for _ in range(10):
		return_boxes, ious = generate_appear_box(im_info, gt_boxes)
		new_im = copy.copy(im)
		for box in return_boxes:
			x1, y1, x2, y2 = box[:4]
			new_im[y1:y2,x1:x2]=1
		print(ious)
		plt.subplot(2,1,2)
		plt.axis('off')
		plt.imshow(new_im)
		plt.title('original')
		plt.show()




