import sys, cv2, os, pickle
sys.path.append('../lib')
import numpy as np
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from utils.timer import Timer
import roi_data_layer.roidb as rdl_roidb
from test_AE_aux import get_image_prepared, pred_box_trans
sys.path.append('../attack_detector')
from block_matrix import block_matrix, save_structure

classes = ('__background__', # always index 0
			 'aeroplane', 'bicycle', 'bird', 'boat',
			 'bottle', 'bus', 'car', 'cat', 'chair',
			 'cow', 'diningtable', 'dog', 'horse',
			 'motorbike', 'person', 'pottedplant',
			 'sheep', 'sofa', 'train', 'tvmonitor')


def pickle_save(save_name, data):
	file = open(save_name,'wb')
	pickle.dump(data, file)
	file.close()
	print('save to {:s}'.format(save_name))


def pickle_load(save_name):
	file = open(save_name,'rb')
	data = pickle.load(file)
	file.close()
	return data 


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

	
def get_clean_nodes_helper(im_set, data_set, sess, net, fetch_list, nodes_dir, sub_dirs):
	
	print('[prepare the dataset...]')
	imdb = get_imdb(data_set) # gt_box, absolute pixel value 0-based from annotation files
	imdb.competition_mode(True) # use_salt: False; cleanup: False
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # ground_truth propose
	rdl_roidb.prepare_roidb(imdb) # get gt rois for training

	assert len(cfg.TEST.SCALES) == 1
	target_size = cfg.TEST.SCALES[0]
	_t = Timer()
	num_images = len(im_set)
	for idx, i in enumerate(im_set):
		#if idx > 15:
		#	break
		i = im_set[idx]
		_t.tic()
		im_cv, im, im_info, gt_boxes = get_image_prepared(cfg,imdb.roidb[idx], target_size)
		feed_dict = {net.data: np.expand_dims(im, axis=0),
					 net.im_info: np.expand_dims(im_info, axis=0),
					 net.appearance_drop_rate: 0.0}

		e_matrix, _, cls_pred,_ = sess.run(fetch_list, feed_dict=feed_dict)
		pred_cls_ids = np.argmax(cls_pred, 1)
		for matrix_idx, sub_dir in enumerate(sub_dirs):
			for node_idx, cls_id in enumerate(pred_cls_ids):
				#if cls_id != 0:
				#	continue
				np_name = '/'.join([nodes_dir.format(classes[cls_id]), sub_dir, 'im{:s}_node{:d}.npy'.format(str(i), node_idx)])
				np.save(np_name, e_matrix[matrix_idx][node_idx])	
		_t.toc()
		print('im_detect: {:d}/{:d} {:.3f}s'\
					.format(idx+1, num_images, _t.average_time))
		

def get_p_nodes_helper(im_set, data_set, sess, net, fetch_list, nodes_dirs, sub_dirs, p_list_dir, p_list_name, p):
	print('[prepare the dataset...]')
	imdb = get_imdb(data_set) # gt_box, absolute pixel value 0-based from annotation files
	imdb.competition_mode(True) # use_salt: False; cleanup: False
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # ground_truth propose
	rdl_roidb.prepare_roidb(imdb) # get gt rois for training

	assert len(cfg.TEST.SCALES) == 1
	target_size = cfg.TEST.SCALES[0]
	_t = Timer()
	perturbations = list(open(os.path.join(p_list_dir, p_list_name)))
	len_ps = len(perturbations)
	
	for print_idx, file_name in enumerate(perturbations):
		#if print_idx > 15:
		#	break
		save_name = file_name.strip().split('/')[-1].strip('.p')
		
		file_id, box_id, f_id, t_id = file_name.strip().split('/')[-1].split('.')[0].split('_')[:4]
		file_id = int(file_id[2:])
		box_id = int(box_id[3:])
		f_id = int(f_id[1:])
		t_id = int(t_id[1:])
		idx, i = im_set.index(file_id), file_id
		_, _, _, gt_boxes = get_image_prepared(cfg, imdb.roidb[idx], target_size)
		_t.tic()
		if True:#for box_id in range(num_gt_boxes):
			cls_id = int(gt_boxes[box_id,-1])
			class_name = imdb._classes[cls_id]
			p_name = os.path.join(p_list_dir, file_name.strip())
			perturbation = block_matrix.load(p_name)
			#print("load perturbation from "+perturbation_name.format(class_name, i, box_id))
			im_cv, im, im_info, gt_boxes = get_image_prepared(cfg,imdb.roidb[idx], target_size, perturbation*p)
			feed_dict = {net.data: np.expand_dims(im, axis=0),
						 net.im_info: np.expand_dims(im_info, axis=0),
						 net.appearance_drop_rate: 0.0}
			e_matrix, rois, cls_pred, bbox_deltas = sess.run(fetch_list, feed_dict=feed_dict)
			# fint the one overlap with gt_box >0.5, test if the attack succeed (prob<0.3), if succeed, save to 
			# corresponding neg dirs.
			pred_boxes = pred_box_trans(rois, cls_pred, bbox_deltas, im_info[-1], im_cv.shape)
			node_overlaps = compare_iou(pred_boxes, gt_boxes[box_id,:4])
			iou_thred = 0.7
			idx_list = np.where(node_overlaps >= iou_thred)[0]
			if len(idx_list) == 0:
				print('\t Shit, no rp boxes for the gt area :(. {:1.2f}'.format(np.max(node_overlaps)))
				_t.toc()
				print('im_detect: {:d}/{:d} {:.3f}s'\
					.format(print_idx+1, len_ps, _t.average_time))
				continue 
			orig_pred_cls_ids = np.argmax(cls_pred, axis=1)
			pred_cls_ids = np.argmax(cls_pred, axis=1)[idx_list] # predictions of the related bboxes
			pred_f = np.where(pred_cls_ids==f_id)[0]
			pred_t = np.where(pred_cls_ids==t_id)[0]
			pred_background = np.where(pred_cls_ids==0)[0]
			cnt_total, cnt_f, cnt_t, cnt_background = len(idx_list), len(pred_f), len(pred_t), len(pred_background)
			cnt_others = cnt_total - cnt_f - cnt_t
			cnt_others = cnt_others if t_id ==0 else cnt_others-cnt_background
			#print('{:d}: {:d}, {:d}, {:d}, {:d}'.format(
			#	cnt_total, cnt_f, cnt_t, cnt_background, cnt_others))
			if cnt_f >0:
				print('\t old class still exists')
				_t.toc()
				print('im_detect: {:d}/{:d} {:.3f}s'\
					.format(print_idx+1, len_ps, _t.average_time))
				continue

			if cnt_t == 0:
				print('\t no target class exists')
				_t.toc()
				print('im_detect: {:d}/{:d} {:.3f}s'\
					.format(print_idx+1, len_ps, _t.average_time))
				continue 	
				
			for idx_name, matrix in enumerate(e_matrix):
				#if sub_dirs[idx_name] not in {'scene_feature', 'object_feature'}:
				#	continue
				if idx_name >= 5:
					continue
				for node_idx in idx_list:
					pred_id = orig_pred_cls_ids[node_idx]
					assert  pred_id!= f_id
					nodes_dir = nodes_dirs.format(classes[pred_id])
					file_to_save = save_name+'_node{:d}.npy'.format(node_idx)
					np_name = '/'.join([nodes_dir, sub_dirs[idx_name], file_to_save])
					np.save(np_name, matrix[node_idx])
					#print(np_name)
		_t.toc()
		print('im_detect: {:d}/{:d} {:.3f}s'\
				.format(print_idx+1, len_ps, _t.average_time))


def get_p_nodes_helper_appear(im_set, data_set, sess, net, fetch_list, nodes_dirs, sub_dirs, p_list_dir, p_list_name, p):
	print('[prepare the dataset...]')
	imdb = get_imdb(data_set) # gt_box, absolute pixel value 0-based from annotation files
	imdb.competition_mode(True) # use_salt: False; cleanup: False
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # ground_truth propose
	rdl_roidb.prepare_roidb(imdb) # get gt rois for training
	assert len(cfg.TEST.SCALES) == 1
	target_size = cfg.TEST.SCALES[0]
	num_matrix = len(sub_dirs)
	_t = Timer()
	perturbations = list(open(os.path.join(p_list_dir, p_list_name)))
	len_ps = len(perturbations)
	
	for print_idx, file_name in enumerate(perturbations):
		#if print_idx > 15:
		#	break
		save_name = file_name.strip().split('/')[-1].strip('.p')
		file_id, box_info, iou_info, t_id = save_name.split('_')[:4]

		file_id = int(file_id[2:])
		box_info = box_info.strip(')').strip('box(')
		x1, y1, x2, y2 = box_info.split('-')
		x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
		box_id = -1 #change for appear
		t_id = int(t_id[1:])
		idx, i = im_set.index(file_id), file_id 
		
		_t.tic()
		if True:
			p_name = os.path.join(p_list_dir, file_name.strip())
			perturbation = block_matrix.load(p_name)
			
			im_cv, im, im_info, gt_boxes = get_image_prepared(cfg,imdb.roidb[idx], target_size, perturbation*p)
			gt_boxes = np.concatenate([gt_boxes, 
					np.expand_dims([x1,y1,x2,y2,0], axis=0)])

			feed_dict = {net.data: np.expand_dims(im, axis=0),
						net.im_info: np.expand_dims(im_info, axis=0),
						net.appearance_drop_rate: 0.0}
			e_matrix, rois, cls_pred, bbox_deltas = sess.run(fetch_list, feed_dict=feed_dict)
			# fint the one overlap with gt_box >0.5, test if the attack succeed (prob<0.3), if succeed, save to 
			# corresponding neg dirs.
			pred_boxes = pred_box_trans(rois, cls_pred, bbox_deltas, im_info[-1], im_cv.shape)
			node_overlaps = compare_iou(pred_boxes, gt_boxes[box_id,:4])
			iou_thred = 0.7
			idx_list = np.where(node_overlaps>=iou_thred)[0]
			if len(idx_list) == 0:
				print('\t Shit, no rp boxes for the gt area :(. {:1.2f}'.format(np.max(node_overlaps)))
				_t.toc()
				print('im_detect: {:d}/{:d} {:.3f}s'\
					.format(print_idx+1, len_ps, _t.average_time))
				continue 
			orig_pred_cls_ids = np.argmax(cls_pred, axis=1)
			pred_cls_ids = np.argmax(cls_pred, axis=1)[idx_list] # predictions of the related bboxes
			pred_t = np.where(pred_cls_ids==t_id)[0]
			pred_background = np.where(pred_cls_ids==0)[0]
			cnt_total,  cnt_t, cnt_background = len(idx_list),  len(pred_t), len(pred_background)
			
			#print('{:d}: {:d},  {:d}'.format(
			#	cnt_total, cnt_t,  cnt_background, ))
			if cnt_t == 0:
				print('\t no appearing class exists')
				_t.toc()
				print('im_detect: {:d}/{:d} {:.3f}s'\
					.format(print_idx+1, len_ps, _t.average_time))
				continue 
			
			for idx_name, matrix in enumerate(e_matrix):
				#if sub_dirs[idx_name] not in {'scene_feature', 'object_feature'}:
				#	continue
				if idx_name >= 5:
					continue
				for node_idx in idx_list:
					pred_id = orig_pred_cls_ids[node_idx]
					nodes_dir = nodes_dirs.format(classes[pred_id])
					file_to_save = save_name+'_node{:d}.npy'.format(node_idx)
					np_name = '/'.join([nodes_dir, sub_dirs[idx_name], file_to_save])
					np.save(np_name, matrix[node_idx])
					#print(np_name)
			
		_t.toc()
		print('im_detect: {:d}/{:d} {:.3f}s'\
				.format(print_idx+1, len_ps, _t.average_time))


	


