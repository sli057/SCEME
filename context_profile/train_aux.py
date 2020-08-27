import os
import numpy as np
import tensorflow as tf
from distutils.version import LooseVersion
from datasets.factory import get_imdb
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from fast_rcnn.bbox_transform import bbox_transform
import datasets.imdb
from roi_data_layer.minibatch import _sample_rois, _project_im_rois


def get_training_roidb(imdb):
	""" returns a roidb (region of interest database)
	for use in training. """
	if cfg.TRAIN.USE_FLIPPED:
		print("Appending horizontally-flipped training examples...")
		imdb.append_flipped_images() 
		# add new rois (flipped boxes) rios are ground truth by annotation.
		# one entry is the rois for one image
		print("Done")

	print("Preparing training data...")
	rdl_roidb.prepare_roidb(imdb)
	"""Enrich the imdb's roidb by adding some derived quantities that
	are useful for training. This function precomputes the maximum
	overlap, taken over ground-truth boxes, between each ROI and each 
	ground-truth box. The class with maximum overlap is also recorded.

	each entry rois[index] =
	{image: image_path,
	width: scaler
	height: scaler
	boxes: [num_box, 4], absolute pixel, 0-based ! no background boxes
	gt_classes: [num_box], the ground-truth class for each box
	gt_overlaps: [num_box, num_class] one-hot verson of gt_classes
	flipped: True/False
	max_classes: exactly the gt_class [num_box] why we have this?
	max_overlaps:  all one vector [num_box] why we have this?
	}
	"""
	print("Done")
	return imdb.roidb


def combined_roidb(imdb_names):
	def get_roidb(imdb_name):
		imdb = get_imdb(imdb_name) # existing dataset object
		print("Loaded dataset {:s} for training".format(imdb.name))
		imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
		roidb = get_training_roidb(imdb)
		return roidb # why not just return imdb?

	roidbs = [get_roidb(s) for s in imdb_names.split('+')]
	roidb = roidbs[0]
	if len(roidbs)>1:
		for r in roidbs[1:]:
			roidb.extend(r)
		imdb = datasets.imdb(imdb_names) # unknown dataset, create new imdb
	else:
		imdb = get_imdb(imdb_names)
	return imdb, roidb 


def filter_roidb(roidb):
	""" remove roidb entries that have no usable ROIs"""
	def is_valid(entry):
		""" valid images have:
		1) at least one foreground ROI or 
		2) at least one background ROI """
		overlaps = entry['max_overlaps']
		fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
		bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI)&
			(overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
		valid = len(fg_inds) + len(bg_inds)
		return valid > 0
		# not make sense to me, aha, reuse for other cases.
	num = len(roidb)
	filtered_roidb = [entry for entry in roidb if is_valid(entry)]
	num_after = len(filtered_roidb)
	print("Filter {} roidb entries: {} --> {}" \
		.format(num-num_after, num, num_after))
	return filtered_roidb


def _compute_targets(rois, overlaps, labels):
	""" compute bounding-box regression targets for an image
	rios [num_box, 4]
	overlappes [num_box]
	labels [num_box] 
	"""
	""" Do not know where max_overlaps get bigger than gt_overlaps,
	becuase previously, max_overlaps = roi[i]['gt_overlaps'].max(axis=1)
	and where soft overlaps are calculated """
	# indices of gt ROIs

	gt_inds = np.where(overlaps == 1)[0] # reuse for other cases
	if len(gt_inds) == 0:
		# bail if the image has no gt ROIs
		return np.zeros((rois.shape[0],5), dtype=np.float32) # the gt for every rois

	# inices of examples for which we try to make predictions
	# Overlap required between a ROI and ground-truth box in order for that ROI to
	# be used as a bounding-box regression training example
	ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

	# get IoU overlap between each exROI and gt ROI
	ex_gt_overlaps = bbox_overlaps(
		np.ascontiguousarray(rois[ex_inds,:], dtype=np.float),
		np.ascontiguousarray(rois[gt_inds,:], dtype=np.float))
	# gt_boxes could overlap? one box is pared with only one gt boxes.
	#[ num_gt_inds, num_ex_inds]
	
	# find which gt ROI each ex ROI has max overlap with:
	# this will be the ex ROI's gt target
	gt_assignment = ex_gt_overlaps.argmax(axis=1)
	gt_rois = rois[gt_inds[gt_assignment],:] 
	ex_rois = rois[ex_inds,:]

	targets = np.zeros((rois.shape[0],5), dtype=np.float32)
	targets[ex_inds,0] = labels[ex_inds] # copy labels, why not ground truth label?
	targets[ex_inds,1:] = bbox_transform(ex_rois, gt_rois) # no anchor boxes? No! not now
	return targets


def add_bbox_regression_targets(roidb):
	""" add information needed to train bounding-box regressors. """
	assert len(roidb) > 0
	assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

	num_images = len(roidb)
	num_classes = roidb[0]['gt_overlaps'].shape[1]
	for im_i in xrange(num_images):
		rois = roidb[im_i]['boxes']
		max_overlaps = roidb[im_i]['max_overlaps']
		max_classes = roidb[im_i]['max_classes']
		roidb[im_i]['bbox_target'] = \
			_compute_targets(rois, max_overlaps, max_classes)
		#[copied label, tx, ty, tw, th]

	if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
		means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes,1))
		stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes,1))
	else:
		raise NotImplementedError(" error")
	# normalize targets
	if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
		for im_i in xrange(num_images):
			targets = roidb[im_i]['bbox_target']
			for cls in xrange(1,num_classes):
				cls_inds = np.where(targets[:,0]==cls)[0]
				roidb[im_i]['bbox_target'][cls_inds, 1:] -= means[cls,:]
				roidb[im_i]['bbox_target'][cls_inds, 1:] /= stds[cls,:]
	return means.ravel(), stds.ravel()


def get_data_layer(roidb, num_classes):
	if cfg.TRAIN.HAS_RPN:
		if cfg.IS_MULTISCALE:
			raise NotImplementedError(" error")
		else:
			layer = RoIDataLayer(roidb, num_classes)
	else:
		raise NotImplementedError(" error")
	return layer


def get_rpn_cls_loss(net):
	"""
	rpn_cls_score = tf.reshape(net.get_output('rpn_cls_score_reshape'),[-1,2])
	rpn_label = tf.reshape(net.get_output('rpn-data')[0],[-1]) 
	rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2]) #(256, 2)
	rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1]) #(256,)
	rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))
	"""
	# ================== RPN  classification loss =============================
	rpn_cls_score = tf.reshape(net.get_output('rpn_cls_score_reshape'),[-1,2]) 
	# [H*W*9,2], object or not
	rpn_label = tf.reshape(net.get_output('rpn-data')[0],[-1])
	# anchor label, [H*W*9], 1 for object, 0 for background, -1 for do not care
	valid_index = tf.where(tf.not_equal(rpn_label,-1))
	rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, valid_index), [-1,2]) #[num_box_reserved, 2]
	rpn_label = tf.reshape( tf.gather(rpn_label, valid_index), [-1]) #[num_box_reserved,2]
	rpn_cross_entropy = tf.reduce_mean(
						tf.nn.sparse_softmax_cross_entropy_with_logits(
							logits = rpn_cls_score,
							labels = rpn_label))
	
	return rpn_cross_entropy


def _modified_smooth_l1(sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
	"""
	ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
	SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
				  |x| - 0.5 / sigma^2,    otherwise
	"""
	sigma2 = sigma * sigma
	inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets)) #[1,1,1,1]
	smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0/sigma2), tf.float32)
	smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5*sigma2)
	smooth_l1_option2 = tf.subtract( tf.abs(inside_mul), 0.5/sigma2)
	smooth_l1_result = tf.add(
		tf.multiply(smooth_l1_option1, smooth_l1_sign),
		tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
	outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)
	return outside_mul


def get_rpn_box_loss(net):
	# ================== RPN bounding box regression L1 loss =================
	rpn_bbox_pred = net.get_output('rpn_bbox_pred') # delta [1,H,W,9*4]
	rpn_bbox_targets = tf.transpose(
						net.get_output('rpn-data')[1],
						[0,2,3,1]) # [1,H,W,9*4] (tx,ty,th,tw) gt for each anchor box
	rpn_bbox_inside_weights = tf.transpose(
						net.get_output('rpn-data')[2],
						[0,2,3,1]) # [1,H,W,9*4]
	rpn_bbox_outside_weights = tf.transpose(
						net.get_output('rpn-data')[3],
						[0,2,3,1])
	rpn_smooth_l1 = _modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, 
					rpn_bbox_inside_weights, rpn_bbox_outside_weights)
	rpn_box_loss = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1,2,3]))
	return rpn_box_loss


def get_RCNN_cls_loss(net):
	cls_score = net.get_output('cls_score')# [num_box, num_class] [128,21]
	label = tf.reshape(net.get_output('roi-data')[1],[-1]) #[num_box, 1] num_box=128
	# change to do misclassification
	#label = tf.gather(label, net.keep_slice)
	#cls_score = tf.gather(cls_score, net.keep_slice)

	cross_entropy = tf.reduce_mean(
					tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
	return cross_entropy


def get_RCNN_box_loss(net):
	bbox_pred = net.get_output('bbox_pred') #[128,84]
	bbox_targets = net.get_output('roi-data')[2] #[num_box, num_class*4]
	bbox_inside_weights = net.get_output('roi-data')[3]
	bbox_outside_weights = net.get_output('roi-data')[4]

	smooth_l1 = _modified_smooth_l1(
				1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
	loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1]))
	return loss_box


def summary(grads_and_vars, loss, cls_loss, box_loss, rpn_cls_loss, rpn_box_loss):
	tf.summary.scalar('loss',loss)
	tf.summary.scalar('cls_loss', cls_loss)
	tf.summary.scalar('box_loss', box_loss)
	tf.summary.scalar('rpn_cls_loss', rpn_cls_loss)
	tf.summary.scalar('rpn_box_loss', rpn_box_loss)

	for var in tf.trainable_variables():
		tf.summary.histogram(var.op.name, var)
	for grad, var in grads_and_vars:
		if grad is not None:
			tf.summary.histogram(var.op.name+'/grad', grad)
	summary_op = tf.summary.merge_all()
	return summary_op


def snapshot(sess, saver, filename, net, bbox_means, bbox_stds):
	if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
		# save original values
		with tf.variable_scope('bbox_pred', reuse=True):
			# tf.get_variable: Gets an existing variable with these parameters or create a new one.
			weights = tf.get_variable("weights")
			biases = tf.get_variable("biases")
		orig_weights = weights.eval(session=sess)
		orig_biases = biases.eval(session=sess)
		# scale and shift with bbox reg unnormalization
		weights_shape = weights.get_shape().as_list()
		feed_dict = {net.bbox_weights: orig_weights * np.tile(bbox_stds,(weights_shape[0],1))}
		sess.run(net.bbox_weights_assign, feed_dict=feed_dict)
		feed_dict = {net.bbox_biases: orig_biases*bbox_stds+bbox_means}
		sess.run(net.bbox_bias_assign, feed_dict=feed_dict)
	# SAVE SNAPCHAT
	saver.save(sess, filename)

	if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
		# restore 
		sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_weights})
		sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_biases})
	return


def sparse_l1_decent(loss, x, q, eps):
	grad, = tf.gradients(loss,x)
	assert len(grad.get_shape()) == 1+3
	red_ind = list(range(1,len(grad.get_shape())))
	dim = tf.reduce_prod(tf.shape(x)[1:])
	abs_grad = tf.reshape(tf.abs(grad), (-1, dim))
	# if q is a scaler, broadcast it to a vector of same length as the batch dim
	q_tile = tf.cast(
		tf.tile(
			tf.expand_dims(q,-1),
			tf.expand_dims(tf.shape(x)[0],-1)),
		tf.float32)

	k = tf.cast(
		tf.floor(q_tile*tf.cast(dim, tf.float32)),
		tf.int32)
	# tf.sort is much faster than tf.contrib.distributions.percentile
	# for TF <= 1.12, use tf.nn.top_k as tf.sort is not implemented.
	if LooseVersion(tf.__version__) <= LooseVersion('1.12.0'):
		sorted_grad = -tf.nn.top_k(-abs_grad, k=dim, sorted=True)[0]
	else:
		sorted_grad = tf.sort(abs_grad, axis=-1)
	idx = tf.stack(
			(tf.range(tf.shape(abs_grad)[0]), k),
			-1)
	percentiles = tf.gather_nd(sorted_grad, idx)
	tied_for_max = tf.greater_equal(abs_grad, tf.expand_dims(percentiles, -1))
	tied_for_max = tf.reshape(
					tf.cast(tied_for_max, x.dtype),
					tf.shape(grad))
	#num_ties = tf.reduce_sum(tied_for_max, red_ind, keepdims=True)
	optimal_perturbation_norm = tf.sign(grad)*tied_for_max#/num_ties
	#eps_scale = eps*tf.to_float(tf.shape(x)[1]*tf.shape(x)[2])*q
	optimal_perturbation = tf.scalar_mul(eps, optimal_perturbation_norm)
	return optimal_perturbation

# start comment
def get_data_layer_self_version(roidb, num_classes):
	if len(roidb) != cfg.TRAIN.IMS_PER_BATCH:
		raise NotImplementedError("please use the true one")
	else:
		print("Just for code test, do not use in real training")
		get_minibatch(roidb, num_classes)

def _sample_rios(riodb, fg_rois_per_image, rois_per_image, num_classes):
	""" generate a random sample of RoIs comprising foreground and background
	examples """
	print("You should not call this unless you are test the code")
	labels = roidb['max_classes'] #[num_box]
	overlaps = roidb['max_overlaps'] #[num_box]
	rois = riodb['boxes'] #[num_box, 4] abosulte pixel value, 0-based
	# select foreground RoIs as those with >= FG_THRESH overlap
	fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
	fg_rois_per_this_image = int(np.minimum(fg_rois_per_image, fg.inds.size))
	if fg_inds.size > 0:
		fg_inds = np.random.choice(fg_inds, size=fg_rois_per_this_image, replace=False)
	# select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI]
	bg_inds = np.where( (overlaps < cfg.TRAIN.BG_THRESH_HI)&
						(overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
	bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
	bg_rois_per_this_image = np.minimum(bg_rois_per_this_image, bg_inds.size) 
	# so what if not enough bg?
	if bg_inds.size > 0:
		bg_inds = np.random.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

	# the indices that we are selecting
	keep_inds = np.append(fg_inds, bg_inds)
	labels = labels[keep_inds] # cls number
	labels[fg_rois_per_this_image:] = 0 # backgorund 
	overlaps = overlaps[keep_inds]
	rois = rois[keep_inds]

	bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
			roidb['bbox_targets'][keep_inds,:], num_classes)
	return label, overlaps, rois, bbox_targets, bbox_inside_weights

def get_minibatch(roidb, num_classes):
	""" given a roidb, construct a minibatch sampled from it 
	roidb : minibatch-db """
	num_images = len(roidb) # cfg.TRAIN.IMS_PER_BATCH = 2
	# sample random scales to use for each image in this batch
	# here just ignore, 'cause we have only one scale
	assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
		'num_images ({}) must divide BATCH_SIZE ({})'. \
		format(num_images, cfg.TRAIN.BATCH_SIZE)
	rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images #128/2 = 64
	fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION*rois_per_image) #0.5*64 =32
	# get the input image blob
	im_blob, im_scales = _get_image_blob(roidb, 0)
	# [num_image, max_height, max_width, 3]
	# [num_iamge, cv2_resize_scale]
	blobs = {'data': im_blob}
	if cfg.TRAIN.HAS_RPN:
		raise NotImplementedError(" error")
	else:
		# now bulid the region of interest and label blobs
		rois_blob = np.zeros((0,5), dtype=np.float32)
		labels_blob = np.zeros((0), dtype=np.float32)
		bbox_targets_blob = np.zeros((0,4*num_classes), dtype=np.float32)
		bbox_inside_blob = np.zeros((bbox_targets_blob.shape), dtype=np.float32)
		for im_i in xrange(num_images):
			labels, overlaps, im_rois, bbox_targets, bbox_inside_weights = \
				_sample_rois(riodb[im_i], fg_rois_per_image, rois_per_image, num_classes)
			# labels [rois_per_image] last half  all 0 (background)
			# overlaps[rois_per_image] 
			# im_rois [rois_per_image, 4]
			# bbox_targets [rois_per_image, 4*K] (tx, ty, tw, th)
			# bbox_inside_weights [ rois_per_image, 4*K] (1,1,1,1) ? what is this?
			# sometimes if not enough boes rois_per_image could not be meet

			# add to rois blob
			rois = _project_im_rois(im_rois, im_scales[im_i])
			batch_ind = im_i * np.ones((roi.shape[0],1)) 
			#! great! problem solved, now how about not enough for batch_size
			rois_blob_this_image = np.hstack((batch_ind, rois))
			rois_blob = np.vstack((rois_blob, rois_blob_this_image))

			# add to labels, bbox targes, and bbox loss blobs
			labels_blob = np.hstack((labels_blob, labels))
			bbox_targets_blob = np.hstack((bbox_targets_blob, bbox_targets))
			bbox_inside_blob = np.vstack((bbox_inside_blob, bbox_inside_weights))

		blobs['rois'] = rois_blob
		blobs['labels'] = labels_blob

		if cfg.TRAIN.BBOX_REG:
			blobs['bbox_targets'] = bbox_targets_blob
			blobs['bbox_inside_weights'] = bbox_inside_blob
			# what is this for?
			blobs['bbox_outside_weights'] = np.array(bbox_inside_blob>0).astype(float32)
	return blobs
# end comment


