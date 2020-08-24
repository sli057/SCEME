## my_test.py
import os, sys, argparse
sys.path.append('../lib')
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import numpy as np
import cv2, cPickle
from fast_rcnn.config import cfg, get_output_dir, cfg_from_file
from datasets.factory import get_imdb
from networks.factory import get_network
from utils.blob import im_list_to_blob
from utils.cython_nms import nms
from utils.timer import Timer
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#from my_get_imdb import get_imdb

def test(args=None):
	parser = argparse.ArgumentParser(description='Simple testing script.')

	parser.add_argument('--net_final', help='the pretrained net', type=str, default='../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wo_context/VGGnet_wo_context.ckpt')
	parser.add_argument('--net_name', help='net_name', type=str, default="VGGnet_wo_context")
	parser.add_argument('--test_set', help='train set', type=str, default="voc_2007_test")
	parser = parser.parse_args(args)

	test_data = parser.test_set 
	net_name = parser.net_name
	net_final = parser.net_final
	if net_final is not None:
		print('varialbes in the pretrained file are:')
		print_tensors_in_checkpoint_file(file_name=net_final, 
									tensor_name='',
									all_tensors = False,
									all_tensor_names = True)

	extra_config = "../experiments/cfgs/faster_rcnn_end2end.yml"
	cfg_from_file(extra_config)
	imdb = get_imdb(test_data)
	imdb.competition_mode(True) # use_salt: False; cleanup: False
	weights_filename = os.path.splitext(os.path.basename(net_final))[0]
	output_dir = get_output_dir(imdb,weights_filename+"_"+net_name)

	# start a session
	net = get_network(net_name+"_test")
	saver = tf.train.Saver()
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)

	saver.restore(sess, net_final)
	print('Loading model weights from {:s}'.format(net_final))
	

	fetch_list = [net.get_output('cls_prob'),
				net.get_output('bbox_pred'),
				net.get_output('rois')]

	assert len(cfg.TEST.SCALES) == 1
	target_size = cfg.TEST.SCALES[0]
	max_per_image = 300
	thresh = 0.05

	num_images = len(imdb.image_index)
	all_boxes = [[[] for _ in xrange(num_images)]
					for _ in xrange(imdb.num_classes)]
	

	_t = {'im_detect':Timer(), 'misc':Timer()}

	#num_images = 200
	for i in xrange(num_images):

		im_cv = cv2.imread(imdb.image_path_at(i))
		im = im_cv.astype(np.float32, copy=True) 
		im -= cfg.PIXEL_MEANS
		im_size_min = np.min(im.shape[0:2])
		im_size_max = np.max(im.shape[0:2])
		im_scale = min([float(target_size) / im_size_min,
						float(cfg.TEST.MAX_SIZE) / im_size_max])
		# the biggest axis should not be more than MAX_SIZE
		im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
					interpolation=cv2.INTER_LINEAR)
		im_info = np.array([im.shape[0],im.shape[1],im_scale],
						dtype=np.float32)
		
		_t['im_detect'].tic()

		feed_dict = {net.data: np.expand_dims(im, axis=0),
					net.im_info: np.expand_dims(im_info, axis=0)}#,
					#net.keep_prob: 1.0}
		

		cls_prob, box_deltas, rois = sess.run(fetch_list,
					feed_dict=feed_dict)

		scores = cls_prob
		boxes = rois[:,1:5] / im_scale # scale first??
		boxes = bbox_transform_inv(boxes, box_deltas)
		boxes = clip_boxes(boxes, im_cv.shape)
		_t['im_detect'].toc()
		_t['misc'].tic()
		# skip j = 0, since it is the background class
		for j in xrange(1, imdb.num_classes):
			inds = np.where(scores[:,j] > thresh)[0] 
			# which boxes has object belongs to class j
			cls_scores = scores[inds,j] # [num_box]
			cls_boxes = boxes[inds,j*4:(j+1)*4] # [num_box,4]
			cls_dets = np.hstack( (cls_boxes, cls_scores[:,np.newaxis])) \
						.astype(np.float32, copy=False) #[num_box, 4 + score]
			keep = nms(cls_dets, cfg.TEST.NMS)
			cls_dets = cls_dets[keep,:]
			all_boxes[j][i] = cls_dets 
			# j class, exists on image i, and have one or more boxes
		# limt to max_per_image detection *over all classes*
		image_scores = np.hstack([all_boxes[j][i][:,-1]
						for j in xrange(1, imdb.num_classes)])
		if len(image_scores) > max_per_image:
			image_thresh = np.sort(image_scores)[-max_per_image]
			for j in xrange(1, imdb.num_classes):
				keep = np.where(all_boxes[j][i][:,-1] >= image_thresh)[0]
				all_boxes[j][i] = all_boxes[j][i][keep,:]
		_t['misc'].toc()
		print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'\
			.format(i+1, num_images, _t['im_detect'].average_time, 
				_t['misc'].average_time))

	det_file = os.path.join(output_dir, "detections.pkl")
	with open(det_file, 'wb') as f:
		cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
	print 'Evaluating detections'
	imdb.evaluate_detections(all_boxes, output_dir)

	print(net_final)
		

if __name__ == '__main__':
	test()







