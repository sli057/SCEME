from __future__ import print_function
import sys, os, argparse
import tensorflow as tf
import numpy as np
sys.path.append('../lib')
sys.path.append('../context_profile')
from fast_rcnn.config import cfg, cfg_from_file
from utils.timer import Timer
from test_AE_aux import  get_image_prepared
from valid_detection import is_valid as is_valid_w_context
from valid_detection_wt_context import is_valid as is_valid_wt_context
from block_matrix import block_matrix, create_mask, create_sticker
from attack_aux import build_physical_adv_graph, prepare_dataset
from visual_p import plt, image_to_plot
from sticker_aux import Stickers
from appear_aux import generate_appear_box
def get_p_box(net, im_cv, im, im_info, gt_boxes, target_id, box_idx, mask, 
	sess, op, varops, placeholders, max_iteration=700, plt_im=False):
	iteration = 0
	feed_dict = { placeholders['PIXEL_MEANS']: cfg.PIXEL_MEANS,
			placeholders['image_in']: np.expand_dims(np.copy(im)*(1-mask), axis=0), #
			# no mean subtract, [#im, H, W, 3] what is the victim set?
			placeholders['noise_mask']: mask, # [H, W, 3]
			net.im_info: np.expand_dims(im_info, axis=0), # need
			net.gt_boxes: gt_boxes, # need 
			net.keep_prob: 1.0}
	while iteration < max_iteration:
		iteration += 1
		#noisy_in = sess.run(varops['noise_inputs'], feed_dict=feed_dict)
		_, loss, pred_loss, smooth_loss, print_loss, noisy_in = sess.run( 
			(op, varops['adv_loss'], varops['predict_loss'], varops['smooth_loss'],
				varops['printer_error'],  varops['noise_inputs']), feed_dict=feed_dict)
		print('		iter {:d}/{:d}: total_loss  {:1.2f} = {:1.2f} + {:1.2f} + {:1.2f}'.format(
			iteration, max_iteration,
			loss, pred_loss, smooth_loss, print_loss), end = '\r')
		if iteration > 200 and pred_loss < 0.39:
			p_im = np.squeeze(noisy_in).astype(np.int32).astype(np.float32)
			if is_valid_w_context(im_cv, p_im-cfg.PIXEL_MEANS, im_info, gt_boxes[box_idx], target_id):#iteration < max_iteration:
				return True, p_im-im

		if plt_im:
			plt.plot()
			plt.imshow(image_to_plot(np.squeeze(noisy_in)))
			plt.pause(0.05)
	p_im = np.squeeze(noisy_in).astype(np.int32).astype(np.float32)
	if is_valid_w_context(im_cv, p_im-cfg.PIXEL_MEANS, im_info, gt_boxes[box_idx], target_id):#iteration < max_iteration:
		return p_im-im
	return None
		

	

def get_p_set(im_set, im_list, save_dir, num_sticker, shape, attack_type, net_name="VGGnet_wt_context",
 				skip_idx=0, max_idx=10*1000):

	stickers = Stickers(num_sticker, shape)
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	suname = str(num_sticker)+shape
	if attack_type == 'appear':
		perturbation_name = '/'.join([save_dir, 'im{:d}_box({:1.0f}-{:1.0f}-{:1.0f}-{:1.0f})_iou{:1.3f}_t{:d}_'+suname])
	else:
		perturbation_name = '/'.join([save_dir,'im{:d}_box{:d}_f{:d}_t{:d}_'+suname])
	
	# some configuration 
	extra_config = "../experiments/cfgs/faster_rcnn_end2end.yml"
	cfg_from_file(extra_config)
	
	# prepare data
	imdb = prepare_dataset(im_set, cfg)
	# prepare graph and sess
	
	num_images = len(im_list)
	_t = Timer()
	for idx, i in enumerate(im_list):
		if idx < skip_idx:
			continue
		if idx > max_idx:
			break
		im_cv, im, im_info, gt_boxes = get_image_prepared(cfg, imdb.roidb[idx], zero_mean=False)
		num_gt_boxes = len(gt_boxes)
		_t.tic()
		for box_id in range(num_gt_boxes):
			valid = is_valid_w_context(im_cv, im-cfg.PIXEL_MEANS, im_info, gt_boxes[box_id], t_id=int(gt_boxes[box_id][-1]))
			if not valid:
				break
		if not valid:
			print("{:d}/{:d}: ignore the image since at least one object is not detected correctly".format(idx+1, num_images))
			continue
		if attack_type == 'appear':
			ori_gt_boxes = gt_boxes
			new_gt_boxes, iou_list = generate_appear_box(im_info, gt_boxes) #(x1, y1, x2, y2, gt_cls=0)
			num_iter = len(new_gt_boxes)
		else:
			num_iter = num_gt_boxes
		
		H, W = int(im_info[0]), int(im_info[1])
		op, net, sess, placeholders, varops = build_physical_adv_graph(net_name=net_name, H=H, W=W)
		for box_id in range(num_iter):
			new_box_id = box_id
			if attack_type == 'appear':
				box_id = -1
				gt_boxes = np.concatenate([ori_gt_boxes, 
					np.expand_dims(new_gt_boxes[new_box_id], axis=0)])
			gt_cls = int(gt_boxes[box_id,-1])
			gt_cls_name = imdb._classes[gt_cls]
			for target_cls, target_cls_name in enumerate(imdb._classes):
				if attack_type == 'hiding' and target_cls != 0:
					continue
				elif target_cls == 0 or target_cls == int(gt_boxes[box_id,-1]):

				save_mask = create_mask(im_info[:2], gt_boxes[box_id,:4])
				sticker_mask = stickers.create_stickers(im_info[:2], gt_boxes[box_id,:4])

				gt_boxes[box_id,-1] = target_cls
				p = get_p_box(net,im_cv, im, im_info, gt_boxes, target_cls, box_id, 
								sticker_mask, sess, op, varops, placeholders)
				gt_boxes[box_id,-1] = gt_cls
				if p is not None:
					if attack_type == 'appear':
						save_name = perturbation_name.format(i, gt_boxes[box_id,0], gt_boxes[box_id,1], 
									gt_boxes[box_id,2], gt_boxes[box_id,3], iou_list[box_id], target_cls)
					else:
						save_name = perturbation_name.format(i, box_id, gt_cls, target_cls)
					p = np.int32(p)
					
					block_matrix.save(save_name,
					p, im_info[:2], gt_boxes[box_id,:4])
					#scipy.sparse.save_npz(, p1)
					#np.save()
					print("\n{:s} --> {:s} succeed."\
						.format(gt_cls_name, imdb._classes[target_cls]))

				else:
					print("\n{:s} --> {:s} failed."\
						.format(gt_cls_name, imdb._classes[target_cls]))
		sess.close()
		tf.reset_default_graph()	
		_t.toc()
		print('{:d}/{:d}: perturbation_generated {:.3f}s'\
			.format(idx+1, num_images, _t.average_time))

def parse_parameter(args=None):
	parser = argparse.ArgumentParser(description='Simple training script.')
	parser.add_argument('--num_sticker', help='number of stickers to add', type=int, default=2)
	parser.add_argument('--shape', help='the shape of the sticker', type=str, default='rectangular')
	parser.add_argument('--skip_idx', help='skip the first few images', type=int, default=0)
	parser.add_argument('--max_idx', help='perturb max_idx number of images', type=int, default=6500)
	parser.add_argument('--attack_type', help='appear, hiding, or miscls', type=str, default='miscls')
	return parser
	
if __name__ == '__main__':
	parser = parse_parameter()
	assert parser.attack_type in ['appear','hiding','miscls']
	# if dataset == 'coco':
	# 	im_set = 'coco_2014_minival'
	# 	im_list = list(open( '../data/coco/annotations/coco_2014_minival.txt','r'))
	# 	im_list = [int(idx.strip().split('_')[-1]) for idx in im_list]
	
	im_set = "voc_2007_test"
	im_list = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_list = [int(idx.strip()) for idx in im_list]
	
	save_dir = 'Physical_p_' + parser.attack_type
	
	get_p_set(im_set, im_list, save_dir, parser.num_sticker, parser.shape,
		attack_type= parser.attack_type,
		skip_idx=parser.skip_idx, max_idx=parser.max_idx)