import numpy as np
import pickle, os, re, sys
sys.path.append('../')
from config import coco_classes, voc_classes
classes = voc_classes


def load_context_files(is_benign, attack, chop=True):
	if is_benign:
		txt_name = '../context_profile/voc_detection_{:s}_p10/test_benign.txt'
	else:
		txt_name = '../context_profile/voc_detection_{:s}_p10/{:s}.txt'.format('{:s}',attack)
	cls_dict = dict()
	for cls_idx, cls_name in enumerate(classes):
		cls_dict[cls_name] = [os.path.basename(path) for path in list(open(txt_name.format(cls_name), 'r'))]
	return cls_dict
	# cls: all files in the set


def load_pred_results(is_benign, attack):
	if is_benign:
		test_result_files = 'voc_reconstruction_error/{:s}_model0.0_test_benign.npy'
	else:
		test_result_files = 'voc_reconstruction_error/{:s}_model0.0_{:s}.npy'.format('{:s}', attack) 
	cls_dict = dict()
	for cls_idx, cls_name in enumerate(classes):
		cls_dict[cls_name] = np.load(test_result_files.format(cls_name))
	return cls_dict


def get_key2preds(is_benign, attack):
	#print('==> get score location(s) for each node/region...')
	key2locations = get_key2predlocations(is_benign=is_benign,attack=attack)
	#print('==> get scores...')
	preds = load_pred_results(is_benign=is_benign, attack=attack)
	#print('==> get socre(s) for each node/region...')
	key2predections = dict()
	for key in key2locations:
		key2predections[key] = []
		for cls_name, file_idx in key2locations[key]:
			key2predections[key].append(preds[cls_name][file_idx])
	return key2predections


def get_key2predlocations(is_benign, attack):
	# regular expression match
	if is_benign:
		re_pattern = 'im[0-9]+_node[0-9]+\.npy'
		chop = '.npy'
	else:
		if 'digi' in attack:
			if 'appear' not in attack:
				re_pattern = 'im[0-9]+_box[0-9]+_f[0-9]+_t[0-9]+_node[0-9]+\.npy'
			else:
				re_pattern = 'im[0-9]+_box\([0-9]+-[0-9]+-[0-9]+-[0-9]+\)_iou[0-9,.]+_t[0-9]+_node[0-9]+\.npy'
		else:
			if 'appear' not in attack:
				re_pattern = 'im[0-9]+_box[0-9]+_f[0-9]+_t[0-9]+_.*?_node[0-9]+\.npy' # physical
			else:
				re_pattern = 'im[0-9]+_box\([0-9]+-[0-9]+-[0-9]+-[0-9]+\)_iou[0-9,.]+_t[0-9]+_.*?_node[0-9]+\.npy'
		chop = '_node'
	prog = re.compile(re_pattern)
	cls_dict = load_context_files(is_benign, attack=attack)
	key2locations = dict()
	for cls_name in cls_dict:
		filenames = cls_dict[cls_name]
		for file_idx, filename in enumerate(filenames):
			#print(filename)
			assert prog.match(filename)# .strip())# want to see
			key = filename.split(chop)[0]
			if key not in key2locations:
				key2locations[key] = []
			key2locations[key].append((cls_name, file_idx))
	return key2locations


def merge_preds(key2predections, is_benign, attack):
	results = []
	num_valid_keys = 0
	if is_benign:
		re_pattern_pos = 'im[0-9]+_node[0-9]+'
		prog_pos = re.compile(re_pattern_pos)
		for key in key2predections:
			assert prog_pos.match(key)
			assert len(key2predections[key])==1
			results.append(key2predections[key][0])
			num_valid_keys += 1 
	else:
		if 'appear' in attack:
			re_pattern_neg = 'im[0-9]+_box\([0-9]+-[0-9]+-[0-9]+-[0-9]+\)_iou[0-9,.]+_t[0-9]+'
		else:
			re_pattern_neg = 'im[0-9]+_box[0-9]+.*'
		prog_neg = re.compile(re_pattern_neg)
		
		for key in key2predections:
			assert prog_neg.match(key)
			if 'hiding' not in attack  and 't0' in key:
				continue 
			elif 'hiding' in attack and 't0' not in key:
				continue
			num_valid_keys += 1 
			results+= [np.max(key2predections[key])]
			#the detection is successful as long as one region of the perturbed object is detected
	print('{:d} valid keys'.format(num_valid_keys))
	return results


def load_preds(is_benign, attack):
	key2preds = get_key2preds(is_benign, attack) # {key:[pred1, pred2,..]}
	preds = merge_preds(key2preds, is_benign,  attack)
	return preds





	