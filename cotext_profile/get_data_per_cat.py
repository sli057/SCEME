## generate_AE.py
## my_train.py
import sys, cv2, os
sys.path.append('../lib')
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.core.protobuf import saver_pb2
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from utils.cython_nms import nms
from networks.factory import get_network
from datasets.factory import get_imdb
from utils.timer import Timer
import roi_data_layer.roidb as rdl_roidb
#import roi_data_layer.roidb as rdl_roidb
from my_train_aux import combined_roidb
from my_train_aux import filter_roidb, add_bbox_regression_targets
from my_train_aux import get_data_layer
from my_train_aux import get_rpn_cls_loss, get_rpn_box_loss
from my_train_aux import get_RCNN_cls_loss, get_RCNN_box_loss
from test_AE_aux import vis_detections, get_image_prepared, pred_box_trans
from S3_attack_rate import pickle_save

classes = ('__background__', # always index 0
			 'aeroplane', 'bicycle', 'bird', 'boat',
			 'bottle', 'bus', 'car', 'cat', 'chair',
			 'cow', 'diningtable', 'dog', 'horse',
			 'motorbike', 'person', 'pottedplant',
			 'sheep', 'sofa', 'train', 'tvmonitor')

def compare_iou(pred_boxes, gt_box):
	"""
	return overlaps [256]
	"""
	gt_area = (gt_box[2]-gt_box[0]+1.) * (gt_box[3]-gt_box[1]+1.)
	if len(pred_boxes) == 0:

		return [] 
	# intersection
	ixmin = np.maximum(pred_boxes[:,0], gt_box[0])
	iymin = np.maximum(pred_boxes[:,1], gt_box[1])
	ixmax = np.minimum(pred_boxes[:,2], gt_box[2])
	iymax = np.minimum(pred_boxes[:,3], gt_box[3])
	iw = np.maximum(ixmax-ixmin+1., 0.)
	ih = np.maximum(iymax-iymin+1., 0.)
	inters = iw*ih 
	# union
	uni = gt_area - inters + \
		(pred_boxes[:,2]-pred_boxes[:,0]+1.)*(pred_boxes[:,3]-pred_boxes[:,1]+1.)
	# IoU
	overlaps = inters/uni 
	return overlaps

def get_p_nodes_helper(im_set, data_set, sess, net, fetch_list, nodes_dirs, sub_dirs, perturbation_name, p):
	
	print('[prepare the dataset...]')
	imdb = get_imdb(data_set) # gt_box, absolute pixel value 0-based from annotation files
	imdb.competition_mode(True) # use_salt: False; cleanup: False
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # ground_truth propose
	rdl_roidb.prepare_roidb(imdb) # get gt rois for training

	assert len(cfg.TEST.SCALES) == 1
	target_size = cfg.TEST.SCALES[0]
	num_matrix = len(sub_dirs)
	_t = Timer()
	num_images = len(im_set)
	for idx, i in enumerate(im_set):
		_, _, _, gt_boxes = get_image_prepared(cfg, imdb.roidb[idx], target_size)
		num_gt_boxes = len(gt_boxes)
		_t.tic()
		num_saved_zero, num_saved_low, num_saved_mis = 0, 0, 0
		for box_id in range(num_gt_boxes):
			cls_id = int(gt_boxes[box_id,-1])
			class_name = imdb._classes[cls_id]
			perturbation = np.load(perturbation_name.format(class_name, i, box_id))
			#print("load perturbation from "+perturbation_name.format(class_name, i, box_id))
			im_cv, im, im_info, gt_boxes = get_image_prepared(cfg,imdb.roidb[idx], target_size, perturbation*p)
			feed_dict = {net.data: np.expand_dims(im, axis=0),
						net.im_info: np.expand_dims(im_info, axis=0)}#,
						#net.appearance_drop_rate: droprate}
			e_matrix, rois, cls_pred, bbox_deltas = sess.run(fetch_list, feed_dict=feed_dict)
			# fint the one overlap with gt_box >0.5, test if the attack succeed (prob<0.3), if succeed, save to 
			# corresponding neg dirs.
			pred_boxes = pred_box_trans(rois, cls_pred, bbox_deltas, im_info[-1], im_cv.shape)
			node_overlaps = compare_iou(pred_boxes, gt_boxes[box_id,:4])
			iou_thred = 0.7
			idx_list = np.where(node_overlaps>=iou_thred)[0]
			if len(idx_list) == 0:
				print('\t Shit, no rp boxes for the gt area :(.')
				continue 
			score_threshold = 0.2
			new_idx_list = np.where(
				np.logical_and(node_overlaps>=iou_thred, cls_pred[:,cls_id] < score_threshold)
				)[0]
			if len(new_idx_list) == 0:
				print('\t Haaa, the attack is not successful.')
				continue
			else:
				print('\t {:d}/{:d} boxes get attacked!'.format(len(new_idx_list),len(idx_list)))

			pred_cls_ids = np.argmax(cls_pred, axis=1)
			for idx_name, matrix in enumerate(e_matrix):
				#if sub_dirs[idx_name] not in {'scene_feature', 'object_feature'}:
				#	continue
				if idx_name < 5:
					continue
				for node_idx in new_idx_list:
					pred_cls_id = pred_cls_ids[node_idx]
					pred_cls_name = classes[pred_cls_id]
					"""
					if pred_cls_id == 0:
						num_saved_zero += 1
						nodes_dir = nodes_dirs[0].format(pred_cls_name)
					"""
					if pred_cls_id == cls_id:
						nodes_dir = nodes_dirs[0].format(pred_cls_name)
						num_saved_low += 1 
					else:
						nodes_dir = nodes_dirs[1].format(pred_cls_name)
						if pred_cls_id == 0:
							num_saved_zero += 1 
						else:
							num_saved_mis += 1 
					np_name = '/'.join([nodes_dir, sub_dirs[idx_name], 'im{:d}_box{:d}_node{:d}.npy'.format(i, box_id, node_idx)])
					np.save(np_name, matrix[node_idx])

		_t.toc()
		print('im_detect: {:d}/{:d} {:.3f}s, #zero{:d} + #low {:d} + #mis {:d}'\
				.format(idx+1, num_images, _t.average_time, num_saved_zero/num_matrix, num_saved_low/num_matrix, num_saved_mis/num_matrix))
	#pickle_save('/'.join([nodes_dir,'category_list'+data_set]), category_list)
	
def get_clean_nodes_helper(im_set, data_set, sess, net, fetch_list, nodes_dir, sub_dirs, perturbation_name, cat_id):
	total = 0
	print('[prepare the dataset...]')
	imdb = get_imdb(data_set) # gt_box, absolute pixel value 0-based from annotation files
	imdb.competition_mode(True) # use_salt: False; cleanup: False
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # ground_truth propose
	rdl_roidb.prepare_roidb(imdb) # get gt rois for training

	assert len(cfg.TEST.SCALES) == 1
	target_size = cfg.TEST.SCALES[0]
	category_list = dict()
	_t = Timer()
	num_images = len(im_set)
	for idx, i in enumerate(im_set):
		i = im_set[idx]
		_, _, _, gt_boxes = get_image_prepared(cfg, imdb.roidb[idx], target_size)
		num_gt_boxes = len(gt_boxes)
		_t.tic()
		im_cv, im, im_info, gt_boxes = get_image_prepared(cfg,imdb.roidb[idx], target_size)
		feed_dict = {net.data: np.expand_dims(im, axis=0),
					net.im_info: np.expand_dims(im_info, axis=0),
					net.appearance_drop_rate: droprate}
		e_matrix, _, cls_pred,_ = sess.run(fetch_list, feed_dict=feed_dict)
		idx_list = np.where(np.argmax(cls_pred, 1) == cat_id)[0]
		for idx_name, matrix in enumerate(e_matrix):
			#if sub_dirs[idx_name] not in {'scene_feature', 'object_feature'}:
			#	continue
			for node_idx in idx_list:
				np_name = '/'.join([nodes_dir, sub_dirs[idx_name], 'im{:s}_node{:d}.npy'.format(str(i), node_idx)])
				np.save(np_name, matrix[node_idx])
				#print('saved in {:s}'.format(np_name))	
		_t.toc()
		print('im_detect: {:d}/{:d} {:.3f}s, number of boxes is {:d}'\
					.format(idx+1, num_images, _t.average_time, len(idx_list)))
		total += len(idx_list)
		#if total >= 500:
		#	break
	
droprate = 0.10
def get_nodes(perturbation_name, set_dirs, sub_dirs, p, cat_id):
	
	train_set_07 = 'voc_2007_train'
	val_set_07 = 'voc_2007_val'
	train_set_12 = 'voc_2012_train'
	val_set_12 = 'voc_2012_val'
	test_set = "voc_2007_test"
	
	net_name =   "VGGnet" 
	net_final = "../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet/VGGnet_fast_rcnn_iter_130000.ckpt"
	
	# some configuration
	extra_config = "../experiments/cfgs/faster_rcnn_end2end.yml"
	cfg_from_file(extra_config)

	print('[build the graph...]')
	net = get_network(net_name+"_test") # not sure now
	saver = tf.train.Saver()
	fetch_list = [net.relation, net.get_output('rois'),net.get_output('cls_prob'),net.get_output('bbox_pred')]

	print('[build the session...]')
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)

	print('[initialize the graph parameter & restore from pre-trained file...]')
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, net_final)
	print('Loading model weights from {:s}'.format(net_final))

	
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/train.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_clean_nodes_helper(im_set, train_set_07, sess, net, fetch_list, set_dirs[0], sub_dirs,
					 perturbation_name, cat_id)
	"""
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/val.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_clean_nodes_helper(im_set, val_set_07, sess, net, fetch_list, set_dirs[0], sub_dirs,
					perturbation_name, cat_id)
	im_set = list(open( '../data/VOCdevkit/VOC2012/ImageSets/Main/train.txt','r'))
	im_set = [idx.strip() for idx in im_set]
	get_clean_nodes_helper(im_set, train_set_12, sess, net, fetch_list, set_dirs[0], sub_dirs,
					 perturbation_name, cat_id)
					 
	
	im_set = list(open( '../data/VOCdevkit/VOC2012/ImageSets/Main/val.txt','r'))
	im_set = [idx.strip() for idx in im_set]
	get_clean_nodes_helper(im_set, val_set_12, sess, net, fetch_list, set_dirs[0], sub_dirs,
					 perturbation_name, cat_id)

	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_clean_nodes_helper(im_set, test_set, sess, net, fetch_list, set_dirs[1], sub_dirs,
					 perturbation_name, cat_id)

	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_p_nodes_helper(im_set, test_set, sess, net, fetch_list, set_dirs[2:], sub_dirs,
					 perturbation_name, p)
	"""









