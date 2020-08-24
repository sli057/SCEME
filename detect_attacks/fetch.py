import os 
import numpy as np 
class Fetch:
	def __init__(self, set_name,
		#root_dir="/home/sli057/Context_AE/SIN/my_return_back/detection_dropout___background___p10/",
		#root_dir="/home/sli057/Context_AE/SIN/my_return_back/detection_dropout_aeroplane_p10/",
		#root_dir="/home/sli057/Context_AE/SIN/my_return_back/detection___background___p10/",
		root_dir,#="/home/sli057/Context_AE/SIN/my_return_back/detection_aeroplane_p10/",
		#root_dir="/Users/Shasha/Dropbox/dataset/detection_aeroplane_p10", 
		transform=None):
		"""
		root_dir(string): dataset dir
		set_name(string): in {'train', 'test_pos', 'test_neg'} 
		fetures (string): in {'node_feature', 'e2e_scores', 'reset_objects', 
							'reset_scene', 'update_objects', 'update_scene'}
		transform: function to do feature extraction
		"""
		if not os.path.isdir(root_dir):
			raise ValueError('[Error] root dir path error, no {:s}.'.format(root_dir))
		#if not set_name in {'train', 'test_pos', 'test_neg', 'val_pos', 'val_neg'}:
		#	raise ValueError('[Error] set name error, no {:s}.'.format(set_name))

		self.features = ['node_feature', 'reset_objects', 'reset_scene', 
					'update_objects', 'update_scene']#, 'object_feature', 'scene_feature','node_feature_updated']# 'e2e_scores',
		self.weights = [1/32.0, 1, 1, 1, 1]#, 1/45.0, 1/45.0, 1/45.0 ]#, 1/45.0]4.0
		self.samples = list(open('/'.join([root_dir, set_name+'.txt'])))
		self.samples = [sample.strip() for sample in self.samples]
		self.label = -1 if 'neg' in set_name else 1
		if 'pos' in set_name or 'train' in set_name:
			sub_dir = 'benign'
		elif 'phy_appear' in set_name:
			sub_dir = 'phy_appear'
		elif 'phy' in set_name and 'phy_appear' not in set_name:
			sub_dir = 'phy_miscls'
		elif 'digi_appear' in set_name:
			sub_dir = 'digi_appear'
		elif 'digi_fgsm' in set_name:
			sub_dir = 'digi_fgsm'
		else:
			sub_dir = 'digi_miscls'
		 
		self.dir = os.path.join(root_dir, 'train' if 'train' in set_name else 'test', 
					sub_dir)
					#'mis_negative' if set_name.split('_')[-1]=='neg' else 'positive')
		self.transform = transform
	def __len__(self):
		return len(self.samples)

	def __getitem__(self, idx):
		data1, data2 = [], np.array([0])
		
		for i, feature in enumerate(self.features):
			data = np.load(os.path.join(self.dir,feature,self.samples[idx]))
			if feature == 'e2e_scores':
				data2 = np.expand_dims(np.sort(data*self.weights[i]), axis=0)		
			else:
				data1.append(data*self.weights[i])

		annot = self.label
		sample = {'data1':np.expand_dims(data1,axis=0), 
				'data2':np.expand_dims(data2,axis=0),
				'annot':annot}

		if self.transform is None:
			return sample
		for  transform_func in self.transform:
			sample = transform_func(sample)
		return sample 
	def get_all(self, num_fetch=None):
		datas, labels = [], []
		num_fetch = len(self.samples) if num_fetch is None else \
					min(num_fetch, len(self.samples))
		for idx in range(num_fetch):
			sample = self.__getitem__(idx)
			datas.append(sample['data'])
			labels.append(sample['annot'])
		return np.array(datas), np.array(labels) 




