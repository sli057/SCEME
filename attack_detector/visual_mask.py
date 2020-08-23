## visual_p.py
import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('../lib')
sys.path.append('../context_profile')
from fast_rcnn.config import cfg, cfg_from_file
from test_AE_aux import  get_image_prepared, pred_box_trans
from config import classes,physical_class_ids
from block_matrix import block_matrix
from attack_aux import build_test_graph, prepare_dataset
from utils.cython_nms import nms

#sess, net, fetch_list = build_test_graph(net_name='VGGnet')
		

def extract_id(name):
	# name like "../physical_p_target/im4_box2_f7_t2[_semi].p"
	name = name.split('/')[-1]
	name = name.split('.')[0]
	name = name.split('_')
	#print(name)
	im_id = int(name[0][2:])
	box_id = int(name[1][3:])
	f_id = int(name[2][1:])
	t_id = int(name[3][1:])
	return im_id, box_id, f_id, t_id


def plot_mask(cfg, imdb, im_list,  p_dir, p_name, save_dir = 'visualization', target_fid=None):
	# imdb_im: the data used for one forward pass prediction for the im.
	# box_idx: the box to observe
	# save_dir: where to save the plot
	im_idx, box_idx, f_id, t_id = extract_id(p_name) 
	if target_fid and f_id != target_fid:
		return
	imdb_im = imdb.roidb[im_list.index(im_idx)]
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	if 'physical' in p_name:
		save_name = os.path.join(save_dir, 'det_bf_aft_im{:d}_box{:d}_f{:d}_t{:d}_physical')
	else:
		save_name = os.path.join(save_dir, 'det_bf_aft_im{:d}_box{:d}_f{:d}_t{:d}_digital')
	

	_, im, _, gt_boxes = get_image_prepared(cfg, imdb_im, zero_mean=False)
	# im: [0,255] np.float, BGR
	# gt_boxes: [None,5], scaled according to image.
	plt.clf()
	plt.subplot(2,1,1)
	plt.imshow(image_to_plot(im))
	plt.title('ground-truth')

	perturbation = get_perturbation(p_dir, p_name)
	
	plt.subplot(2,1,2)
	plt.imshow(image_to_plot(im+perturbation))
	plt.title('prediction on perturbed image')

	plt.pause(1)



def get_perturbation(p_dir, p_name):
	p_name = os.path.join(p_dir, p_name)
	return block_matrix.load(p_name)
	

def image_to_plot(im):
	im_to_plot = np.array(np.clip(im,0,255), dtype=np.uint8)
	return im_to_plot[:,:,(2,1,0)]



if __name__ == '__main__':
	# some configuration 
	extra_config = "../experiments/cfgs/faster_rcnn_end2end.yml"
	cfg_from_file(extra_config)
	print('[prepare the dataset...]')
	im_set = "voc_2007_test"
	imdb = prepare_dataset(im_set, cfg)
	im_list = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_list = [int(idx.strip()) for idx in im_list]
	p_dir = 'Physical_p_target'
	physical_mis = 'script_extract_file/physical_miscls.txt'
	file_list = list(open(physical_mis, 'r'))
	for target_fid in physical_class_ids:
		for name in file_list:
			plot_mask(cfg, imdb, im_list, p_dir, name.strip(), target_fid=target_fid)

