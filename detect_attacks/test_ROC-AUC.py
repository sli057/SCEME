import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from test_aux import load_preds
import pickle


def calculate_roc_auc(attack):
	print('==> prepare benign samples')
	benign_preds = load_preds(is_benign=True, attack=attack)
	print('==> prepare pt samples')
	pt_preds = load_preds(is_benign=False, attack=attack)
	
	
	preds = np.concatenate((benign_preds, pt_preds))
	labels = [0]*len(benign_preds) + [1]*len(pt_preds)

	fpr, tpr, thres = metrics.roc_curve(labels, preds, pos_label=1)
	"""
	for idx, cur_fpr in enumerate(fpr, start=0):
		if cur_fpr < 0.11:
			print('fpr: {:1.3f} at thershold {:1.4f}'.format(cur_fpr, thres[idx]))
	"""
	roc_auc = metrics.auc(fpr, tpr)
	print('{:s}: ROC-AUC = {:1.3f}'.format(attack, roc_auc))

	
if __name__ == '__main__':
	attacks = [	'test_digi_ifgsm_miscls',
				'test_digi_ifgsm_hiding',
				'test_digi_ifgsm_appear',
				'test_physical_miscls',
				'test_physical_hiding',
				'test_physical_appear' ]
	for attack in attacks:
		calculate_roc_auc(attack)




