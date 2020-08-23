## VGGnet_train.py
import tensorflow as tf
from networks.network import Network
import numpy as np
n_classes = 21#12
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

def get_whole_box(im_info):
    print "========================= whole box layer ==================="
    scene = [[0, 0, 0, im_info[0], im_info[1]]]
    #union_boxes = np.array(union_boxes).astype(np.float32)
    #scene = np.array(scene).astype(np.float32)
    return scene

class VGGnet_classifier(Network):
	def __init__(self, trainable=True, data=None):
		self.inputs = []
		if data is None:
			self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3]) #[batch_size, height, width, channel]
		else:
			self.data = data
		self.im_info = tf.placeholder(tf.float32, shape=[None, 3]) #[batch_size, height, width, im_scale]
		self.gt_label = tf.placeholder(tf.int32, shape=[None]) #[batch_size, tx, ty, th, tw, cls]
		#self.keep_prob = tf.placeholder(tf.float32)

		self.layers = {'data': self.data,
						'im_info': self.im_info,
						'gt_label': self.gt_label}
		self.trainable = trainable
		self.keep_slice = tf.placeholder(tf.int32)
		self.setup()
		"""
		# create ops and placeholders for bbox normalization process
		with tf.variable_scope('bbox_pred', reuse=True):
			# tf.get_variable: Gets an existing variable with these parameters or create a new one.
			weights = tf.get_variable("weights")
			biases = tf.get_variable("biases")
			self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
			self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())
			self.bbox_weights_assign = weights.assign(self.bbox_weights)
			self.bbox_bias_assign = biases.assign(self.bbox_biases)
		"""


	def setup(self):
		# ===============  VGG backbone =====================
		(self.feed('data')
			.conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
			.conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
			.max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
			.conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
			.conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
			.max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
			.conv(3, 3, 256, 1, 1, name='conv3_1', trainable=True)
			.conv(3, 3, 256, 1, 1, name='conv3_2', trainable=True)
			.conv(3, 3, 256, 1, 1, name='conv3_3', trainable=True)
			.max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
			.conv(3, 3, 512, 1, 1, name='conv4_1', trainable=True)
			.conv(3, 3, 512, 1, 1, name='conv4_2', trainable=True)
			.conv(3, 3, 512, 1, 1, name='conv4_3', trainable=True)
			.max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
			.conv(3, 3, 512, 1, 1, name='conv5_1', trainable=True)
			.conv(3, 3, 512, 1, 1, name='conv5_2', trainable=True)
			.conv(3, 3, 512, 1, 1, name='conv5_3', trainable=True))

		# ===================   RPN ============================
		(self.feed('conv5_3')
			.conv(3, 3, 512, 1, 1, name='rpn_conv/3x3', trainable=True))
		"""
		(self.feed('rpn_conv/3x3')
			.conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False,
				name='rpn_bbox_pred', trainable=True))
		
		(self.feed('rpn_conv/3x3')
			.conv(1,1,len(anchor_scales)*3*2, 1, 1, padding='VALID', relu = False, 
				name='rpn_cls_score', trainable=True)
			.reshape_layer(2, name = 'rpn_cls_score_reshape')
			.softmax(name='rpn_cls_prob')
			.reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))
		# proposed rois
		(self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
			.proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))

		# assign anchors to gt targets, and corresponding labels and regression targets
		# score only to give size info
		# use to construct loss function for the RPN part
		(self.feed('rpn_cls_score','gt_boxes','im_info','data')
			.anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' )) 

		# match proposed rois to gt targets, and get correspoding labels and regression targets
		# use for future detection, the input for RCNN
		(self.feed('rpn_rois','gt_boxes')
			.proposal_target_layer(n_classes, name = 'roi-data'))
		"""
		#========= Union Box ==========

		#self.layers['whole_box'] = get_whole_box(self.im_info)
		
		(self.feed('im_info')
	     .whole_box_layer(name='whole_box'))

		# ======================= R-CNN =========================
		(self.feed('conv5_3', 'whole_box')
	     .roi_pool(7, 7, 1.0/16, name='whole_pool')
			.fc(4096, name='fc6', trainable=True))	

		(self.feed('fc6')
			.fc(n_classes, relu=False, name='cls_score')
			.softmax(name='cls_prob'))
		"""
		(self.feed('fc6')
			.fc(n_classes*4, relu=False, name='bbox_pred'))
		"""



