import os 
import numpy as np 
set_names = ['train_benign', 
				'test_benign', 
				'test_digi_ifgsm_miscls', 
				'test_digi_ifgsm_hiding',
				'test_digi_ifgsm_appear',
				'test_physical_miscls',
				'test_physical_hiding',
				'test_physical_appear']


class Fetch:
	def __init__(self, set_name, root_dir):
		"""
		root_dir(string): dataset dir
		set_name(string): in {'train', 'test_pos', 'test_neg'} 
		"""
		if not os.path.isdir(root_dir):
			raise ValueError('[Error] root dir path error, no {:s}.'.format(root_dir))
		if not set_name in set_names:
			raise ValueError('[Error] set name error, no {:s}.'.format(set_name))

		self.features = ['node_feature', 'reset_objects', 'reset_scene', 
					'update_objects', 'update_scene']#, 'object_feature', 'scene_feature','node_feature_updated']# 'e2e_scores',
		self.weights = [1/32.0, 1, 1, 1, 1]#, 1/45.0, 1/45.0, 1/45.0 ]#, 1/45.0]4.0
		self.samples = list(open(os.path.join(root_dir, set_name+'.txt')))
		self.samples = [sample.strip().split('/')[-1] for sample in self.samples]
		self.label = -1 if 'benign' in set_name else 1
		sub_dir1, sub_dir2 = set_name.split('_')[0], '_'.join(set_name.split('_')[1:])
		self.dir = os.path.join(root_dir, sub_dir1, sub_dir2)

	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		data =[]
		for i, feature in enumerate(self.features):
			tmp = np.load(os.path.join(self.dir,feature,self.samples[idx]))
			data.append(tmp*self.weights[i])

		annot = self.label
		sample = {'data':np.expand_dims(data,axis=0),
				'annot':annot}
		return sample 

	def get_all(self, num_fetch=None):
		data, labels = [], []
		num_fetch = len(self.samples) if num_fetch is None else \
					min(num_fetch, len(self.samples))
		for idx in range(num_fetch):
			sample = self.__getitem__(idx)
			data.append(sample['data'])
			labels.append(sample['annot'])
		return np.array(data), np.array(labels) 




