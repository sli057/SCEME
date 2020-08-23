# orignal SIN, may modified somehow
import tensorflow as tf
from networks.network import Network


n_classes = 81
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
	def __init__(self, trainable=True):
		self.inputs = []
		self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
		self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
		self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
		
		self.appearance_drop_rate = tf.placeholder(tf.float32) ## critical change

		self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
		self.trainable = trainable
		self.relation = []
		self.setup()

		# create ops and placeholders for bbox normalization process
		
		with tf.variable_scope('bbox_pred', reuse=True):
			weights = tf.get_variable("weights")
			biases = tf.get_variable("biases")

			self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
			self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

			self.bbox_weights_assign = weights.assign(self.bbox_weights)
			self.bbox_bias_assign = biases.assign(self.bbox_biases)
		
	def setup(self):
		(self.feed('data')
			 .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
			 .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
			 .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
			 .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
			 .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
			 .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
			 .conv(3, 3, 256, 1, 1, name='conv3_1', trainable=False)
			 .conv(3, 3, 256, 1, 1, name='conv3_2', trainable=False)
			 .conv(3, 3, 256, 1, 1, name='conv3_3', trainable=False)
			 .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
			 .conv(3, 3, 512, 1, 1, name='conv4_1', trainable=False)
			 .conv(3, 3, 512, 1, 1, name='conv4_2', trainable=False)
			 .conv(3, 3, 512, 1, 1, name='conv4_3', trainable=False)
			 .max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
			 .conv(3, 3, 512, 1, 1, name='conv5_1', trainable=False)
			 .conv(3, 3, 512, 1, 1, name='conv5_2', trainable=False)
			 .conv(3, 3, 512, 1, 1, name='conv5_3', trainable=False))

		#=============================== RPN ===============================
		(self.feed('conv5_3')
			.conv(3,3,512,1,1,name='rpn_conv/3x3', trainable=False))
		
		(self.feed('rpn_conv/3x3')
			.conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, 
				name='rpn_bbox_pred', trainable=False))

		(self.feed('rpn_conv/3x3')			 
			.conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, 
				name='rpn_cls_score', trainable=False)
			.reshape_layer(2,name = 'rpn_cls_score_reshape')
			.softmax(name='rpn_cls_prob')
			.reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))

		# RoI Proposal 
		(self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
			 .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))
		# assign anchors to gt targets, and corresponding labels and regression targets
		# score only to give size info
		# use to construct loss function for the RPN part
		(self.feed('rpn_cls_score','gt_boxes','im_info','data')
			 .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

		(self.feed('rpn_rois','gt_boxes')
			 .proposal_target_layer(n_classes,name = 'roi-data'))
	
		#=============================== roi pooling ============================
		(self.feed('roi-data', 'im_info')
			 .union_box_layer(name='whole_box'))

		(self.feed('roi-data', 'im_info')
			.edge_box_layer(name='edges'))	
		
		(self.feed('conv5_3', 'whole_box')
			.roi_pool(7, 7, 1.0/16, name='whole_pool'))
		
		(self.feed('conv5_3', 'roi-data')
			.roi_pool(7, 7, 1.0/16, name='pool_5'))

		# ============================  detection head ===========================
		(self.feed('pool_5','whole_pool')
			.concat(axis=0, name='concat')
			.fc(4096, name='fc6', trainable=True)) 

		# context graph modeling
		(self.feed('fc6', 'edges')
		 	.structure_inference_spmm(self.relation, boxes=128, name='inference', droprate=self.appearance_drop_rate))

		(self.feed('inference')
			.fc(n_classes, relu=False, name='cls_score')
			.softmax(name='cls_prob'))
		 
		(self.feed('inference')
			.fc(n_classes*4, relu=False, name='bbox_pred'))
	