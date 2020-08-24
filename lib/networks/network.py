import numpy as np
import tensorflow as tf
from GRU import GRUCell
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
from rpn_msr.union_box_layer_tf import union_box_layer as union_box_layer_py
from rpn_msr.union_box_layer_tf import whole_box_layer as whole_box_layer_py
from rpn_msr.edge_box_layer_tf import edge_box_layer as edge_box_layer_py


DEFAULT_PADDING = 'SAME'


def layer(op):
	def layer_decorated(self, *args, **kwargs):
		# Automatically set a name if not provided.
		name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
		# Figure out the layer inputs.
		if len(self.inputs)==0:
			raise RuntimeError('No input variables found for layer %s.'%name)
		elif len(self.inputs)==1:
			layer_input = self.inputs[0]
		else:
			layer_input = list(self.inputs)
		# Perform the operation and get the output.
		layer_output = op(self, layer_input, *args, **kwargs)
		# Add to layer LUT.
		self.layers[name] = layer_output
		# This output is now the input for the next layer.
		self.feed(layer_output)
		# Return self for chained calls.
		return self
	return layer_decorated

class Network(object):
	def __init__(self, inputs, trainable=True):
		self.inputs = []
		self.layers = dict(inputs)
		self.trainable = trainable
		self.setup()

	def setup(self):
		raise NotImplementedError('Must be subclassed.')

	def load(self, data_path, session, saver, ignore_missing=False):
		if data_path.endswith('.ckpt'):
			saver.restore(session, data_path)
		else:
			data_dict = np.load(data_path,allow_pickle=True).item()
			for key in data_dict:
				with tf.variable_scope(key, reuse=True):
					for subkey in data_dict[key]:
						try:
							var = tf.get_variable(subkey)
							session.run(var.assign(data_dict[key][subkey]))
							print "assign pretrain model "+subkey+ " to "+key
						except ValueError:
							print "ignore "+key
							if not ignore_missing:
								raise

	def feed(self, *args):
		assert len(args)!=0
		self.inputs = []
		for layer in args:
			if isinstance(layer, basestring):
				try:
					layer = self.layers[layer]
					print layer
				except KeyError:
					print self.layers.keys()
					raise KeyError('Unknown layer name fed: %s'%layer)
			self.inputs.append(layer)
		return self

	def get_output(self, layer):
		try:
			layer = self.layers[layer]
		except KeyError:
			print self.layers.keys()
			raise KeyError('Unknown layer name fed: %s'%layer)
		return layer

	def get_unique_name(self, prefix):
		id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
		return '%s_%d'%(prefix, id) # conv_1

	def make_var(self, name, shape, initializer=None, trainable=True):
		return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)

	def validate_padding(self, padding):
		assert padding in ('SAME', 'VALID')

	@layer
	def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
		# kernel size [k_h, kw]
		# #input_chaneel --> ci; #output_channel -->c0;
		# slide [s_h, s_w]
		self.validate_padding(padding)
		c_i = input.get_shape()[-1]
		assert c_i%group==0
		assert c_o%group==0
		convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
		with tf.variable_scope(name) as scope:

			init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
			init_biases = tf.constant_initializer(0.0)
			kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
			biases = self.make_var('biases', [c_o], init_biases, trainable)

			if group==1:
				conv = convolve(input, kernel)
			else:
				input_groups = tf.split(3, group, input)
				kernel_groups = tf.split(3, group, kernel)
				output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
				conv = tf.concat(3, output_groups)
			if relu:
				bias = tf.nn.bias_add(conv, biases)
				return tf.nn.relu(bias, name=scope.name)
			return tf.nn.bias_add(conv, biases, name=scope.name)

	@layer
	def relu(self, input, name):
		return tf.nn.relu(input, name=name)

	@layer
	def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.max_pool(input,
							  ksize=[1, k_h, k_w, 1],
							  strides=[1, s_h, s_w, 1],
							  padding=padding,
							  name=name)

	@layer
	def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
		self.validate_padding(padding)
		return tf.nn.avg_pool(input,
							  ksize=[1, k_h, k_w, 1],
							  strides=[1, s_h, s_w, 1],
							  padding=padding,
							  name=name)

	@layer
	def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
		# only use the first input
		if isinstance(input[0], tuple):
			input[0] = input[0][0]

		if isinstance(input[1], tuple):
			input[1] = input[1][0]

		print input
		return roi_pool_op.roi_pool(input[0], input[1],
									pooled_height,
									pooled_width,
									spatial_scale,
									name=name)[0]

	@layer
	def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
		if isinstance(input[0], tuple):
			input[0] = input[0][0]
		return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)
		#tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32])


	@layer
	def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
		if isinstance(input[0], tuple):
			input[0] = input[0][0]

		with tf.variable_scope(name) as scope:

			rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32])

			rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
			rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
			rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
			rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')


			return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
			# rpn_labels [1,H,W,9] --> [1,1, 9*H, W]
			# rpn_bbox_targets [1, 9*4, height, width]
			# rpn_bbox_inside_weights [1, 9*4, height, width]
			# rpn_bbox_outside_weights [1, 9*4, height, width]


	@layer
	def proposal_target_layer(self, input, classes, name):
		if isinstance(input[0], tuple):
			input[0] = input[0][0]
		with tf.variable_scope(name) as scope:

			rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights = tf.py_func(proposal_target_layer_py,[input[0],input[1],classes],[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])

			rois = tf.reshape(rois,[-1,5] , name = 'rois') 
			labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
			bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
			bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
			bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

		   
		return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
		# rois [num_box, 5]
		# labels [num_box, 1]
		# bbox_targets  [num_box, num_class*4]
		# bbox_inside_weights  [num_box, num_class*4]
		# bbox_outside_weights  [num_box, num_class*4]
	#=================================================================================================================#

	@layer
	def union_box_layer(self, input, name):
		print "========================= union box layer ==================="
		if isinstance(input[0], tuple):
			input[0] = input[0][0]
	   
		with tf.variable_scope(name) as scope:
			whole = tf.py_func(union_box_layer_py, [input[0], input[1]],[tf.float32])
			whole = tf.reshape(whole, [-1, 5], name = 'whole')
			return whole

	@layer
	def edge_box_layer(self, input, name):
		print "========================= edge box layer ==================="
		if isinstance(input[0], tuple):
			input[0] = input[0][0]
		
		with tf.variable_scope(name) as scope:
			edge = tf.py_func(edge_box_layer_py, [input[0], input[1]],[tf.float32])
			edge = tf.reshape(edge, [-1, 12], name = 'edge')
			return edge

	@layer
	def crop_pool_layer(self, input, name):
		print "========================== crop pool layer ================="
		rois = input[1]
		bottom = input[0]
		with tf.variable_scope(name) as scope:
			batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
			# Get the normalized coordinates of bounding boxes
			bottom_shape = tf.shape(bottom)
			height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(16)
			width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(16)
			x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
			y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
			x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
			y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
			# Won't be back-propagated to rois anyway, but to save time
			bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
			pre_pool_size = 7 * 2
			crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

		return slim.max_pool2d(crops, [2, 2], padding='SAME')
	#=================================================================================================================#


	@layer
	def reshape_layer(self, input, d,name):
		input_shape = tf.shape(input)
		if name == 'rpn_cls_prob_reshape':
			 return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
					int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
		else:
			 return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
					int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

	@layer
	def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
		return feature_extrapolating_op.feature_extrapolating(input,
							  scales_base,
							  num_scale_base,
							  num_per_octave,
							  name=name)

	@layer
	def lrn(self, input, radius, alpha, beta, name, bias=1.0):
		return tf.nn.local_response_normalization(input,
												  depth_radius=radius,
												  alpha=alpha,
												  beta=beta,
												  bias=bias,
												  name=name)

	@layer
	def concat(self, input, axis, name):
		inputs = [input[0], input[1]]
		return tf.concat(values=inputs, axis=axis, name=name)

	@layer
	def fc(self, input, num_out, name, relu=True, trainable=True):
		print "========== fc_layer ========="
		with tf.variable_scope(name) as scope:
			# only use the first input
			if isinstance(input, tuple):
				input = input[0]
	
			print input	
			input_shape = input.get_shape()#[512,49]
			if input_shape.ndims == 4:
				dim = 1
				for d in input_shape[1:].as_list():
					dim *= d
				feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
			else:
				feed_in, dim = (input, int(input_shape[-1]))

			if name == 'bbox_pred':
				init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
				init_biases = tf.constant_initializer(0.0)
			else:
				init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
				init_biases = tf.constant_initializer(0.0)

			weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
			biases = self.make_var('biases', [num_out], init_biases, trainable)

			op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
			fc = op(feed_in, weights, biases, name=scope.name)
			print fc
			return fc

	@layer
	def softmax(self, input, name):
		input_shape = tf.shape(input)
		if name == 'rpn_cls_prob':
			return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
		else:
			return tf.nn.softmax(input,name=name)

	@layer
	def dropout(self, input, keep_prob, name):
		#print "============ drop out ==========="
		return tf.nn.dropout(input, keep_prob, name=name)

	#======================= my edit ================================
	@layer
	def whole_box_layer(self, input, name):
		print "========================= whole box layer ==================="
	   
		with tf.variable_scope(name) as scope:
			whole = tf.py_func(whole_box_layer_py, [input[0]],[tf.float32])
			whole = tf.reshape(whole, [-1, 5], name = 'whole')
			return whole
			
	
	@layer
	def structure_inference_spmm(self, input, relation, boxes, name, droprate=tf.constant(0.0)):
		print "================ structural inference ==============="
		print input  
		
		n_steps = 1# change for coco
		n_boxes = boxes #train 128, test 256
		n_inputs = 4096 #edit D

		n_hidden_o = 4096
		n_hidden_e = 4096
	

		ofe = input[1]
		ofo, ofs = tf.split(input[0], [n_boxes, 1], 0)
		# drop out on ofo ## critical changes
		ofo = tf.nn.dropout(ofo, keep_prob=1-droprate)

		fo = tf.reshape(ofo, [n_boxes, n_inputs])
		fs = tf.reshape(ofs, [1, n_inputs])

		fs = tf.concat(n_boxes * [fs], 0)
		fs = tf.reshape(fs, [n_boxes, 1, n_inputs])

		fe = tf.reshape(ofe, [n_boxes * n_boxes, 12])

		u = tf.get_variable('u', [12, 1], initializer = tf.contrib.layers.xavier_initializer())
		# Form 1
		#W = tf.get_variable('CW', [n_inputs, n_inputs], initializer = tf.orthogonal_initializer())
		
		# Form 2
		Concat_w = tf.get_variable('Concat_w', [n_inputs * 2, 1], initializer = tf.contrib.layers.xavier_initializer())

		E_cell = GRUCell(n_hidden_e)
		O_cell = GRUCell(n_hidden_o)

		PE = tf.nn.relu(tf.reshape(tf.matmul(fe, u), [n_boxes, n_boxes]))
		oinput = fs[:, 0, :]
		hi = fo # current hidden state
		relation.append(fo)
		for t in range(n_steps):
			X = tf.concat(n_boxes * [hi], 0)
			X = tf.reshape(X, [n_boxes * n_boxes, n_inputs])
			Y = hi  # Y = fo: 128 * 4096
				
			# VE form 1:
			# VE = tf.nn.tanh(tf.matmul(tf.matmul(Y, W), tf.transpose(Y))) # Y*W*Y_T = (128 * 4096) * (4096 * 4096) * (4096 * 128) = 128 * 128
		
			# VE form 2:
			Y1 = tf.concat(n_boxes * [Y], 1)
			Y1 = tf.reshape(Y1, [n_boxes * n_boxes, n_inputs])

			Y2 = tf.concat(n_boxes * [Y], 0)
			Y2 = tf.reshape(Y2, [n_boxes * n_boxes, n_inputs])
	
			VE = tf.nn.tanh(tf.reshape(tf.matmul(tf.concat([Y1, Y2], 1), Concat_w), [n_boxes, n_boxes]))
		
			E = tf.multiply(PE, VE)
			Z = tf.nn.softmax(E)    # edge relationships
			X = tf.reshape(X, [n_boxes, n_boxes, n_inputs]) # Nodes
			M = tf.reshape(Z, [n_boxes, n_boxes, 1]) * X # messages
			M = tf.reduce_max(M, 1) # intergated message
		
			me_max = tf.reshape(M, [n_boxes, 1, n_inputs])
			einput = me_max[:, 0, :]

			with tf.variable_scope('o_gru', reuse=(t!=0)):
				ho1, hi1, ro, uo = O_cell(inputs = oinput, state = hi)
		
			with tf.variable_scope('e_gru', reuse=(t!=0)):
				ho2, hi2, re, ue = E_cell(inputs = einput, state = hi)
		
			#maxpooling
			#hi = tf.maximum(hi1, hi2)
				
			#meanpooling
			hi = tf.concat([hi1, hi2], 0)
			hi = tf.reshape(hi, [2, n_boxes, n_inputs])
			hi = tf.reduce_mean(hi, 0) #[n_box, n_inputs] [256,4096]
			if t==0:
				relation.extend([ro, uo, re, ue, Z, hi, oinput, einput])
		return hi
	

