import sys
sys.path.append('../lib')
from config import voc_classes

import os
import shutil

classes = voc_classes

im_set = list(open( '../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt','r'))
im_set = [int(idx.strip()) for idx in im_set]

gen_net_name = "VGGnet_wt_context"
test_net_name = "VGGnet"

file_name = '{:s}'
perturbation_dir = os.path.join("perturbations", gen_net_name, sub_dir_name, file_name)
for class_name in classes:
	if not os.path.exists(perturbation_dir.format(class_name)):
		os.makedirs(perturbation_dir.format(class_name))
	print(perturbation_dir.format(class_name))
perturbation_name = os.path.join(perturbation_dir, 
					'im{:d}_box{:d}.npy') #im_idx


from get_data_all_cat import get_nodes

p_check = 10
root_dir = 'voc_detection_{:s}_p{:d}'.format('{:s}',p_check)
set_dirs= [os.path.join(root_dir, 'train', 'positive'),
			os.path.join(root_dir, 'test', 'positive'),
			os.path.join(root_dir, 'test', 'digi_fgsm_miscls'),
			os.path.join(root_dir, 'test', 'digi_fgsm_hiding'),
			os.path.join(root_dir, 'test', 'digi_fgsm_appear'),
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

get_nodes(perturbation_name, set_dirs, sub_dirs, p_check)











