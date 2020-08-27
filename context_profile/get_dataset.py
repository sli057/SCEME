import shutil
import os, glob
import numpy as np


def parse_files(dir, file_path):
	if not os.path.isdir(dir):
		with open(file_path, 'w') as f:
			f.write('')
		return 0 
	file_list = glob.glob(os.path.join(dir,'*.npy'))
	with open(file_path, 'w') as f:
		for filename in file_list:
			f.write(filename+'\n')
	return len(file_list)


def split_trainval(in_file, out_file_train, out_file_val, ratio=0.05, copy=True):
	if copy:
		shutil.copyfile(in_file, in_file.replace('.txt', 'orig.txt'))
	f = list(open(in_file,'r'))
	f1 = open(out_file_train,'w')
	f2 = open(out_file_val,'w')
	total_entries, test_ens, val_ens = len(f), 0, 0
	for line in f:
		if np.random.uniform(0.0,1.0) < ratio:
			f2.write(line)
			val_ens += 1 
		else:
			f1.write(line)
			test_ens += 1 
	f1.close()
	f2.close()
	print('{:s}: {:d} = {:d} + {:d}'.format(in_file, total_entries, test_ens, val_ens))
	return


def get_dataset(classes, root_dir, set_dirs, sub_dirs):
	set_names = ['train_benign.txt', 
				'test_benign.txt', 
				'test_digi_ifgsm_miscls.txt', 
				'test_digi_ifgsm_hiding.txt',
				'test_digi_ifgsm_appear.txt',
				'test_physical_miscls.txt',
				'test_physical_hiding.txt',
				'test_physical_appear.txt']
	set_names = [ os.path.join(root_dir, name) for name in set_names]
	assert len(set_names) == len(set_dirs)
	sub_dir = sub_dirs[-1]
	for cls_name in classes:
		for set_idx in range(len(set_names)):
			dir_name = os.path.join(set_dirs[set_idx], sub_dir).format(cls_name)
			set_name = set_names[set_idx].format(cls_name)
			cnt = parse_files(dir_name, set_name)
			if 'train_benign' in set_name:
				in_file = set_name 
				out_file_train = set_name
				out_file_val = set_name.replace('train_benign.txt', 'val_benign.txt')
				split_trainval(in_file, out_file_train, out_file_val)
			else:
				print('{:s} get {:d} entries'.format(set_name, cnt))



