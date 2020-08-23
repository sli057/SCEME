import sys
sys.path.append('../lib')
import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.core.protobuf import saver_pb2
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir, get_output_tb_dir
from networks.factory import get_network
from train_aux import combined_roidb
from train_aux import filter_roidb, add_bbox_regression_targets
from train_aux import get_data_layer
from train_aux import get_rpn_cls_loss, get_rpn_box_loss
from train_aux import get_RCNN_cls_loss, get_RCNN_box_loss
from train_aux import summary, snapshot
import os 

def optimistic_restore(session, save_file):
	reader = tf.train.NewCheckpointReader(save_file)
	saved_shapes = reader.get_variable_to_shape_map()
	var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
						if var.name.split(':')[0] in saved_shapes])
	restore_vars = []
	with tf.variable_scope('', reuse=True):
		for var_name, saved_var_name in var_names:
			curr_var = tf.get_variable(saved_var_name)
			var_shape = curr_var.get_shape().as_list()
			if var_shape == saved_shapes[saved_var_name]:
				restore_vars.append(curr_var)
	saver = tf.train.Saver(restore_vars)
	saver.restore(session, save_file)

def train(args=None):
	parser = argparse.ArgumentParser(description='Simple training script.')
	parser.add_argument('--net_pretrained', help='the pretrained net', type=str, default='../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wo_context/VGGnet_fast_rcnn_iter_130000.ckpt')
	parser.add_argument('--net_name', help='net_name', type=str, default="VGGnet")
	parser.add_argument('--train_set', help='train set', type=str, default="coco_2014_train+coco_2014_valminusminival")
	parser.add_argument('--iter_start', help='skip the first few iterations, relates to checkpoint', type=int, default=0)
	parser.add_argument('--max_iters', help='max number of iterations to run', type=int, default=350000)
	parser.add_argument('--droprate', help='skip the first few iterations, relates to checkpoint', type=float, default=0.2)

	parser = parser.parse_args(args)
	net_name = parser.net_name
	droprate = parser.droprate
	pretrained_model = parser.net_pretrained
	iter_start = parser.iter_start
	max_iters = parser.max_iters

	# some configuration
	extra_config = "../experiments/cfgs/faster_rcnn_end2end.yml"
	cfg_from_file(extra_config)
	np.random.seed(cfg.RNG_SEED)

	# get dataset ready!
	print("[preparing the dataset...]")
	imdb, roidb = combined_roidb(parser.train_set)
	"""
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
	#roidb = filter_roidb(roidb)
	if cfg.TRAIN.BBOX_REG:
		# calculate roidb.roidb[im_id].['bbox_target'] [label, tx, ty, tw, th]
		# calculate mean and std, and apply on (tx, ty, tw, th)
		print(" computing bounding-box regression targets ...")
		bbox_means, bbox_stds = add_bbox_regression_targets(roidb)
		print("Done")
	data_layer = get_data_layer(roidb, imdb.num_classes) # for db+db, num_class=0!
	# a layer used to get next mini batch

	# build the graph
	print("[build the graph...]")
	net = get_network(net_name+"_train")
	rpn_cls_loss = get_rpn_cls_loss(net)
	rpn_box_loss = get_rpn_box_loss(net)
	RCNN_cls_loss = get_RCNN_cls_loss(net)
	RCNN_box_loss = get_RCNN_box_loss(net)
	loss = rpn_cls_loss + rpn_box_loss + RCNN_cls_loss + RCNN_box_loss
	saver = tf.train.Saver(max_to_keep=10, write_version=saver_pb2.SaverDef.V1)
	# Saver restores all variables already defined in your model
	# so if do not want to saver momontum, just define saver before that.
	# tf.train.Saver constructor adds save and restore ops to the graph for all, 
	# or a specified list, of the variables in the graph. 
	# The Saver object provides methods to run these ops, 
	# specifying paths for the checkpoint files to write to or read from.
	# meta file: describes the saved graph structure, includes GraphDef, SaverDef, and so on
	# 
	global_step = tf.Variable(0, trainable=False)
	lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, 
									cfg.TRAIN.STEPSIZE, 0.1, staircase=True)
	momentum = cfg.TRAIN.MOMENTUM 
	optimizer = tf.train.MomentumOptimizer(lr, momentum)
	train_op = optimizer.minimize(loss, global_step=global_step)
	fetch_list = [net.get_output('conv1_1'), rpn_cls_loss, rpn_box_loss, RCNN_cls_loss, RCNN_box_loss, train_op]
	gvs = optimizer.compute_gradients(loss)
	summary_op = summary(gvs, loss, RCNN_cls_loss, RCNN_box_loss, rpn_cls_loss, rpn_box_loss)

	print("[trainable variables are:]")  
	for var in tf.trainable_variables():    
		 #var_names.append(var.name)
		print(var)

	# build the session
	print("[build the session and set model&summary save helper]")  
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)
	# api to save model and save summary
	output_dir = get_output_dir(imdb, net_name+"_my_{:1.3f}".format(droprate)) 
	tb_dir = get_output_tb_dir(imdb, net_name+"_my_{:1.3f}".format(droprate))
	train_writer = tf.summary.FileWriter(tb_dir, sess.graph)
	# restore the model
	print("[initalize the graph paremater & restore from pre-trained file")
	sess.run(tf.global_variables_initializer())
	#  Note that when you restore variables you do not have to initialize them beforehand.
	# but we need initialization for lr, global_step, momentum, and others that are not restored.
	if pretrained_model is not None:
		#print('varialbes in the pretrained file are:')
		#print_tensors_in_checkpoint_file(file_name=pretrained_model, 
		#							tensor_name='',
		#							all_tensors = False,
		#							all_tensor_names = True)
		optimistic_restore(sess, pretrained_model)

	for iter in xrange(iter_start, max_iters):
		blobs = data_layer.forward()
		feed_dict = {net.data: blobs['data'], #[num_image, max_height, max_width, 3]
					net.im_info: blobs['im_info'], #[height, width, im_scale]
					net.gt_boxes: blobs['gt_boxes'], #[x1, y1, x2, y2, cls]
					net.appearance_drop_rate: droprate}
					
		"""
		print("***** new iteration *****")
		print(np.shape(blobs['data']))
		print(blobs['data'][0,0,0])
		print(blobs['im_info'])
		print(blobs['gt_boxes'])
		"""

		if (iter+1) % 1000 == 0:
			summary_w = sess.run(summary_op, feed_dict)
			train_writer.add_summary(summary_w, iter)

			filename = (net_name + '_iter_{:d}'.format(iter+1) + '.ckpt')
			filename = os.path.join(output_dir, filename)
			snapshot(sess, saver, filename, net, bbox_means, bbox_stds)
			print("Wrote snapshot to: {:s}".format(filename))

		a, rpn_cls_loss_value, rpn_box_loss_value, RCNN_cls_loss_value, RCNN_box_loss_value, _ = \
			sess.run(fetch_list, feed_dict=feed_dict)
		total_loss = rpn_cls_loss_value + rpn_box_loss_value + RCNN_cls_loss_value + RCNN_box_loss_value
		"""
		print(np.shape(a))
		print(a[0,0,0])
		"""

		if (iter+1) % cfg.TRAIN.DISPLAY == 0:
			print('iter %d/%d: total loss is %.4f. (rpn_cls_loss: %.4f, rpn_box_loss: %.4f, \
				RCNN_cls_loss: %.4f, RCNN_box_loss: %.4f)'% (iter+1, max_iters, total_loss, \
					rpn_cls_loss_value, rpn_box_loss_value, RCNN_cls_loss_value, RCNN_box_loss_value))

	print(" finished ")


if __name__ == '__main__':
	train()


