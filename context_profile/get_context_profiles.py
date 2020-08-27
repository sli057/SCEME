import sys, os, shutil
import tensorflow as tf
sys.path.append('../lib')
from config import voc_classes
from get_nodes_info import get_clean_nodes_helper, get_p_nodes_helper, get_p_nodes_helper_appear
from fast_rcnn.config import cfg_from_file
from networks.factory import get_network
from get_dataset import get_dataset


def set_up_folders(classes):
	p_check = 10
	root_dir = 'voc_detection_{:s}_p{:d}'.format('{:s}',p_check)
	set_dirs= [os.path.join(root_dir, 'train', 'benign'),
				os.path.join(root_dir, 'test', 'benign'),
				os.path.join(root_dir, 'test', 'digi_ifgsm_miscls'),
				os.path.join(root_dir, 'test', 'digi_ifgsm_hiding'),
				os.path.join(root_dir, 'test', 'digi_ifgsm_appear'),
				os.path.join(root_dir, 'test', 'physical_miscls'),
				os.path.join(root_dir, 'test', 'physical_hiding'),
				os.path.join(root_dir, 'test', 'physical_appear')]
				
	obj_feature = 'node_feature'
	reset_scene = 'reset_scene'
	update_scene = 'update_scene'
	reset_objects = 'reset_objects'
	update_objects = 'update_objects'

	sub_dirs = [obj_feature, reset_scene, update_scene, reset_objects, update_objects]

	for cls_idx, cls_name in enumerate(classes, start=0):
		for set_dir in set_dirs:
			for sub_dir in sub_dirs:
				dir_name = os.path.join(set_dir.format(cls_name), sub_dir)
				#if 'digi_appear' in dir_name:
				#	shutil.rmtree(dir_name)
				if not os.path.exists(dir_name):
					os.makedirs(dir_name)
	return root_dir, set_dirs, sub_dirs


def get_context_profiles(set_dirs, sub_dirs, p_scale=1):
	train_set_07 = 'voc_2007_train'
	val_set_07 = 'voc_2007_val'
	train_set_12 = 'voc_2012_train'
	val_set_12 = 'voc_2012_val'
	test_set = "voc_2007_test"

	net_name = "VGGnet"
	net_final = '../output/faster_rcnn_end2end/voc_2007_trainval+voc_2012_trainval/VGGnet_wt_context/VGGnet_wt_context.ckpt'

	# some configuration
	extra_config = "../experiments/cfgs/faster_rcnn_end2end.yml"
	cfg_from_file(extra_config)

	print('[build the graph...]')
	net = get_network(net_name+"_test") 
	saver = tf.train.Saver()
	fetch_list = [net.relation, net.get_output('rois'),net.get_output('cls_prob'),net.get_output('bbox_pred')]

	print('[build the session...]')
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)

	print('[initialize the graph parameter & restore from pre-trained file...]')
	sess.run(tf.global_variables_initializer())
	saver.restore(sess, net_final)
	print('Loading model weights from {:s}'.format(net_final))

	# Note that is is not necessary to collect all the context profiles, just stop the running
	# if you have got enough training/testing samples.

	im_set = list(open('../data/VOCdevkit/VOC2007/ImageSets/Main/train.txt', 'r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_clean_nodes_helper(im_set, train_set_07, sess, net, fetch_list, set_dirs[0], sub_dirs)
	
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/val.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_clean_nodes_helper(im_set, val_set_07, sess, net, fetch_list, set_dirs[0], sub_dirs)
	
	im_set = list(open( '../data/VOCdevkit/VOC2012/ImageSets/Main/train.txt','r'))
	im_set = [idx.strip() for idx in im_set]
	get_clean_nodes_helper(im_set, train_set_12, sess, net, fetch_list, set_dirs[0], sub_dirs)
	
	im_set = list(open( '../data/VOCdevkit/VOC2012/ImageSets/Main/val.txt','r'))
	im_set = [idx.strip() for idx in im_set]
	get_clean_nodes_helper(im_set, val_set_12, sess, net, fetch_list, set_dirs[0], sub_dirs)

	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_clean_nodes_helper(im_set, test_set, sess, net, fetch_list, set_dirs[1], sub_dirs)

	p = 1
	p_list_dir =  '../attack_detector/script_extract_files'
	p_list_name = 'digital_miscls.txt'
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_p_nodes_helper(im_set, test_set, sess, net, fetch_list, set_dirs[2], sub_dirs,
					 p_list_dir, p_list_name, p=p)

	p_list_dir =  '../attack_detector/script_extract_files'
	p_list_name = 'digital_hiding.txt'
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_p_nodes_helper(im_set, test_set, sess, net, fetch_list, set_dirs[3], sub_dirs,
					 p_list_dir, p_list_name, p=p)

	p_list_dir =  '../attack_detector/script_extract_files'
	p_list_name = 'digital_appear.txt'
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_p_nodes_helper_appear(im_set, test_set, sess, net, fetch_list, set_dirs[4], sub_dirs,
					 p_list_dir, p_list_name, p=p)

	p_list_dir =  '../attack_detector/script_extract_files'
	p_list_name = 'physical_miscls.txt'
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_p_nodes_helper(im_set, test_set, sess, net, fetch_list, set_dirs[5], sub_dirs,
					 p_list_dir, p_list_name, p=p)

	p_list_dir =  '../attack_detector/script_extract_files'
	p_list_name = 'physical_hiding.txt'
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_p_nodes_helper(im_set, test_set, sess, net, fetch_list, set_dirs[6], sub_dirs,
					 p_list_dir, p_list_name, p=p)

	p_list_dir =  '../attack_detector/script_extract_files'
	p_list_name = 'physical_appear.txt'
	im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
	im_set = [int(idx.strip()) for idx in im_set]
	get_p_nodes_helper_appear(im_set, test_set, sess, net, fetch_list, set_dirs[7], sub_dirs,
					 p_list_dir, p_list_name, p=p)


if __name__ == '__main__':
	root_dir, set_dirs, sub_dirs = set_up_folders(voc_classes)
	get_context_profiles(set_dirs, sub_dirs)
	get_dataset(voc_classes, root_dir, set_dirs, sub_dirs)












