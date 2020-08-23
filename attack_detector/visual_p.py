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
from config import classes
from block_matrix import block_matrix
from attack_aux import build_test_graph, prepare_dataset
from utils.cython_nms import nms

sess, net, fetch_list = build_test_graph(net_name='VGGnet')

def prediction(imdb_im, cfg, p=None, thresh=0.5):
	im_cv, im, im_info, gt_boxes = get_image_prepared(cfg, imdb_im, perturbations=p)
	feed_dict = {net.data: np.expand_dims(im, axis=0),
				net.im_info: np.expand_dims(im_info, axis=0)}
	cls_prob, box_deltas, rois = sess.run(fetch_list, feed_dict=feed_dict)
	pred_boxes = pred_box_trans(rois, cls_prob, box_deltas, im_info[-1], im_cv.shape) 
	# scaled boxes on scaled im
	scores = cls_prob #[num_box, num_class]
	boxes = pred_boxes
	# skip j = 0, since it is the background class
	all_boxes = np.zeros([0,25])
	for j in xrange(1, len(classes)):
		inds = np.where(scores[:,j] > thresh)[0] 
		# which boxes has object belongs to class j
		pred_scores = scores[inds,j]
		cls_dets = np.hstack( (boxes[inds], pred_scores[:,np.newaxis])) \
					.astype(np.float32, copy=False) #[num_box, 4 + score]
		dets = np.hstack( (boxes[inds], scores[inds])) \
					.astype(np.float32, copy=False) #[num_box, 4 + score]
		if len(cls_dets) == 0:
			continue
		keep = nms(cls_dets, cfg.TEST.NMS)
		all_boxes = np.vstack([all_boxes,dets[keep,:]])
	return all_boxes 
		

def extract_id(name):
	# name like "../physical_p_target/im4_box2_f7_t2[_semi].p"
	name = name.split('/')[-1]
	name = name.split('.')[0]
	name = name.split('_')
	print(name)
	im_id = int(name[0][2:])
	box_id = int(name[1][3:])
	f_id = int(name[2][1:])
	t_id = int(name[3][1:])
	return im_id, box_id, f_id, t_id


def plot_det(cfg, imdb, im_list,  p_dir, p_name, save_dir = 'visualization'):
	# imdb_im: the data used for one forward pass prediction for the im.
	# box_idx: the box to observe
	# save_dir: where to save the plot
	im_idx, box_idx, f_id, t_id = extract_id(p_name) 
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
	plt.subplot(2,2,1)
	plt.imshow(image_to_plot(im))
	vis_detections(plt, gt_boxes)
	plt.title('ground-truth')


	perturbation = get_perturbation(p_dir, p_name)
	plt.subplot(2,2,2)
	plot_p = perturbation if 'physical' in p_dir else perturbation*10
	plt.imshow(image_to_plot(plot_p))
	vis_detections(plt, [gt_boxes[box_idx]])
	plt.title('perturbation')


	plt.subplot(2,2,3)
	plt.imshow(image_to_plot(im))
	pt_boxes = prediction(imdb_im, cfg, p=None)
	vis_detections(plt, pt_boxes)
	plt.title('prediction on clean image')

	plt.subplot(2,2,4)
	plt.imshow(image_to_plot(im+perturbation))
	pt_boxes = prediction(imdb_im, cfg, p=perturbation)
	vis_detections(plt, pt_boxes)
	plt.title('prediction on perturbed image')

	plt.savefig(save_name.format(im_idx, box_idx, f_id, t_id))
	plt.pause(0.3)
	print('saved figure in '+save_name.format(im_idx, box_idx, f_id, t_id))

def vis_detections(plt, dets, thred=0.2):
	num_dets = len(dets)
	for i in range(num_dets):
		det = dets[i]
		box = det[:4]
		if len(det) == 5:
			class_idx = int(det[-1])
			score = None 
		else:
			assert len(det[4:])==len(classes)
			class_idx = np.argmax(det[4:])
			score = np.max(det[4:])
		if score and score < thred:
			continue 
		plt.gca().add_patch(
			plt.Rectangle(
				(box[0], box[1]), 
				box[2]-box[0], box[3]-box[1],
				fill = False, edgecolor='g', linewidth=3))
		label = classes[class_idx] 
		if score:
			label += ' {:.3f}'.format(score)
		plt.gca().text(box[0], box[1]-2, label,
			bbox=dict(facecolor='blue', alpha=0.5),
			fontsize=6, color='white')
	return 


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
	p_dir = 'physical_p_target'
	physical_mis = 'script_extract_file/physical_miscls.txt'
	file_list = list(open(physical_mis, 'r'))
	for name in file_list:
		plot_det(cfg, imdb, im_list, p_dir, name.strip())

