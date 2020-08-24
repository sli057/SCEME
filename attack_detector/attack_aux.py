import sys,os
sys.path.append('../lib')
sys.path.append('../context_profile')
from datasets.factory import get_imdb
import roi_data_layer.roidb as rdl_roidb
from networks.factory import get_network
from train_aux import get_rpn_cls_loss, get_rpn_box_loss
from train_aux import get_RCNN_cls_loss, get_RCNN_box_loss
import tensorflow as tf
import numpy as np

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def prepare_dataset(im_set, cfg):
	print('[prepare the dataset...]')
	imdb = get_imdb(im_set) 
	# gt_box, absolute pixel value 0-based from annotation files
	imdb.competition_mode(True) # use_salt: False; cleanup: False
	imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD) # ground_truth propose
	rdl_roidb.prepare_roidb(imdb) # get gt rois for training
	#orig_cls = imdb._class_to_ind[orig_cls]
	#target_cls = imdb._class_to_ind[target_cls]
	return imdb 


def build_test_graph(net_name):

	if net_name ==  "VGGnet":
		net_final = '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wt_context/VGGnet_wt_context.ckpt'
		#net_final = '../output/faster_rcnn_end2end/coco_2014_train+coco_2014_valminusminival/VGG_context_maxpool_cp/VGGnet_context_maxpool_iter_113000.ckpt'
	elif net_name == "VGGnet_wo_context":
		net_final = '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wo_context/VGGnet_wo_context.ckpt'
		#net_final = '../output/faster_rcnn_end2end/coco_2014_train/VGGnet_wt_context_version2/VGGnet_wt_context_iter_350000.ckpt'
	else:
		raise ValueError("net {:s} does not implimented")
	print('[build the graph...]')
	g = tf.Graph()
	with g.as_default():
		with HiddenPrints():
			net = get_network(net_name+"_test") 
		saver = tf.train.Saver()
		fetch_list = [net.get_output('cls_prob'),
				net.get_output('bbox_pred'),
				net.get_output('rois')]

	print('[build the session...]')
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config, graph=g)

	print('[initialize the graph parameter & restore from pre-trained file...]')
	#sess.run(tf.global_variables_initializer())
	saver.restore(sess, net_final)
	print('Loading model weights from {:s}'.format(net_final))
	return sess, net, fetch_list



def build_physical_adv_graph(net_name='VGGnet_wt_context', H=None, W=None,
	targeted=True,  print_opt=True, noise_clip=False,
	noisy_input_clip_min=0.0, noisy_input_clip_max=255.0,
	lr=0.5, adam_beta1=0.9, adam_beta2=0.999, adam_epslion=1e-8):
	with HiddenPrints():
		print('[build the session & graph...]')
		sess = tf.Session()
		#H, W = 32,32
		placeholders = {}
		placeholders['PIXEL_MEANS'] = tf.placeholder(tf.float32, shape=(1, 1, 3))
		placeholders['image_in'] = tf.placeholder(tf.float32, shape=(None, H, W, 3)) # no mean subtract
		# [#im, H, W, 3] what is the victim set?
		placeholders['noise_mask'] = tf.placeholder(tf.float32, shape=(H, W, 3))
		# [H, W, 3]
		if print_opt:
			placeholders['printable_colors'] = tf.constant(get_print_triplets(H,W,src_file='npstriplets.txt'))
			#[#colors, H, W, 3]
		varops = {}
		#H, W = tf.shape(placeholders['noise_mask'])[0], tf.shape(placeholders['noise_mask'])[1]
		#varops['noise'] = tf.Variable(tf.random_uniform([H, W, 3], 0, 1), # initial
		#	name='noise', collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'])
		varops['noise'] = tf.Variable(np.ones([H, W, 3],dtype=np.float), # initial
			name='noise', collections = [tf.GraphKeys.GLOBAL_VARIABLES, 'adv_var'],
			dtype=tf.float32)

		if noise_clip:
			varops['noise'] = tf.clip_by_value(varops['noise'], noise_clip_min, noise_clip_max)
		varops['mask_area']=tf.reduce_sum(placeholders['noise_mask'])/3
		varops['noise_mul']=tf.multiply(placeholders['noise_mask'], varops['noise'])

		varops['noise_inputs'] = tf.clip_by_value(tf.add(placeholders['image_in'], varops['noise_mul']), 
										noisy_input_clip_min, noisy_input_clip_max) 
		sticker_vars = set(tf.global_variables()) # only noise variable
		
		net = get_network(net_name+"_train", varops['noise_inputs']- placeholders['PIXEL_MEANS']) # not sure now
		model_vars = set(tf.global_variables())-sticker_vars # only Faster-RCNN  varialbe
		saver = tf.train.Saver(var_list = model_vars)
		if net_name ==  "VGGnet":
			net_final = '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wt_context/VGGnet_wt_context.ckpt'
			#net_final = '../output/faster_rcnn_end2end/coco_2014_train+coco_2014_valminusminival/VGG_context_maxpool_cp/VGGnet_context_maxpool_iter_113000.ckpt'
		elif net_name == "VGGnet_wo_context":
			net_final = '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wo_context/VGGnet_wo_context.ckpt'
			#net_final = '../output/faster_rcnn_end2end/coco_2014_train/VGGnet_wt_context_version2/VGGnet_wt_context_iter_350000.ckpt'
		else:
			raise ValueError("net {:s} does not implimented")
		saver.restore(sess, net_final)
		print('Loading model weights from {:s}'.format(net_final))
		rpn_cls_loss = get_rpn_cls_loss(net)
		rpn_box_loss = get_rpn_box_loss(net)
		RCNN_cls_loss = get_RCNN_cls_loss(net)
		RCNN_box_loss = get_RCNN_box_loss(net)
		# !!!! here optimizer not miminize loss, the gradient is to maximum loss! 
		# every sign should be opposite!!
		loss = rpn_cls_loss + rpn_box_loss + RCNN_box_loss
		if targeted:
			loss += RCNN_cls_loss 
		else:
			raise ValueError("Please re-consider your choice")
			loss += - RCNN_cls_loss
		varops['predict_loss'] = loss
		# control the smoothness of the perturbation
		varops['smooth_loss'] = tf.reduce_sum(tf.image.total_variation(varops['noise']))/varops['mask_area']/1000# use noise_inputs if do not blank sticker area at first
		if print_opt:
			varops['printab_pixel_element_diff'] = tf.squared_difference(varops['noise']/255.0, placeholders['printable_colors']) #change to [0,255] # use noise_inputs if do not blank sticker area at first
			varops['printab_pixel_diff'] = tf.sqrt(tf.reduce_sum(varops['printab_pixel_element_diff'], 3))
			varops['printab_reduce_prod'] = tf.reduce_prod(varops['printab_pixel_diff'], 0)
			varops['printer_error'] = tf.reduce_sum(varops['printab_reduce_prod'])/varops['mask_area']
			varops['adv_loss'] =  varops['predict_loss']  + varops['smooth_loss'] + varops['printer_error']#
		else:
			varops['adv_loss'] = varops['predict_loss'] #+ varops['smooth_loss']

		#op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(varops['adv_loss'], 
		#	var_list = tf.get_collection('adv_var'))
		op = tf.train.AdamOptimizer(learning_rate=lr, beta1=adam_beta1, beta2=adam_beta2,
			epsilon=adam_epslion).minimize(varops['adv_loss'], var_list = tf.get_collection('adv_var'))
		#gvs = optimizer.compute_gradients(varops['adv_loss'], var_list = tf.get_collection('adv_var'))
		#clip_gvs =[(tf.sign(var), var) for grad, var in gvs]
		#op = optimizer.apply_gradients(clip_gvs)
		sess.run(tf.variables_initializer(set(tf.global_variables())-model_vars)) # noise varialbe + opt variable
	return op, net, sess, placeholders, varops


	"""
	feed_dict = {
		placeholders['PIXEL_MEANS']: tf.placeholder(tf.float, shape=(3)),
		placeholders['image_in']: tf.placeholder(tf.float32, shape=(None, None, None, 3)), 
		# no mean subtract, [#im, H, W, 3] what is the victim set?
		placeholders['noise_mask']: tf.placeholder(tf.float32, shape=(None, None, 3)),
		# [H, W, 3]
		placeholders['printable_colors']: tf.placeholder(tf.float32, shape=(None, None, None, 3)),	
		net.im.info: np.expand_dims(im_info, axis=0), # need
		net.gt_boxes: gt_boxes, # need 
		net.keep_prob: 1.0}
	"""
	
def get_print_triplets(H, W, src_file='npstriplets.txt'):
	'''
	Reads the printability triplets from the specified file
	and returns a numpy array of shape (num_triplets, FLAGS.img_cols, FLAGS.img_rows, nb_channels)
	where each triplet has been copied to create an array the size of the image
	:param src: the source file for the printability triplets
	:return: as described 
	'''
	p = []
	# load the triplets and create an array of the speified size
	with open(src_file) as f:
		for l in f:
			p.append(l.split(","))
	p = map(lambda x: [[x for _ in xrange(W)] for __ in xrange(H)], p)
	p = np.float32(p) #transformed to int
	#p = np.int32(np.float32(p)*255)
	return p#np.float32(p)

	


def build_digital_adv_graph(net_name, targeted=True):
	print('[build the graph...]')
	g = tf.Graph()
	with g.as_default():
		with HiddenPrints():
			net = get_network(net_name+"_train") 
		saver = tf.train.Saver()
		rpn_cls_loss = get_rpn_cls_loss(net)
		rpn_box_loss = get_rpn_box_loss(net)
		RCNN_cls_loss = get_RCNN_cls_loss(net)
		RCNN_box_loss = get_RCNN_box_loss(net)
		# !!!! here optimizer not miminize loss, the gradient is to maximum loss! 
		# every sign should be opposite!!
		loss = rpn_cls_loss + rpn_box_loss + RCNN_box_loss
		if targeted:
			loss += RCNN_cls_loss 
		else:
			raise ValueError("Please re-consider your choice")
			loss += - RCNN_cls_loss
		grad, = tf.gradients(-loss, net.data) #default maximize
		grad = tf.sign(grad)

	print('[build the session...]')
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config, graph=g)
	#sess.run(tf.global_variables_initializer())
	if net_name ==  "VGGnet":
		net_final = '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wt_context/VGGnet_wt_context.ckpt'
		#net_final = '../output/faster_rcnn_end2end/coco_2014_train+coco_2014_valminusminival/VGG_context_maxpool_cp/VGGnet_context_maxpool_iter_113000.ckpt'
	elif net_name == "VGGnet_wo_context":
		net_final = '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wo_context/VGGnet_wo_context.ckpt'
		#net_final = '../output/faster_rcnn_end2end/coco_2014_train/VGGnet_wt_context_version2/VGGnet_wt_context_iter_350000.ckpt'
	else:
		raise ValueError("net {:s} does not implimented")
	saver.restore(sess, net_final)
	print('Loading model weights from {:s}'.format(net_final))
	return sess, net, grad

if __name__ == '__main__':
	p=get_print_triplets(2, 2)
	print(p)