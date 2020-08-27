import torch
from torch.utils.data import DataLoader
from fetch import Fetch
from auto_encoder import AutoEncoder

import numpy as np 
import argparse
import os 
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from config import voc_classes
classes = voc_classes


def plot_roc(args=None):
	parser = argparse.ArgumentParser(description='Simple training script.')
	parser.add_argument('--cls_id', help='cls name', type=int)
	parser.add_argument('--versions', help='model version', type=float, nargs='*')
	parser.add_argument('--saves_dir', help='roc preds path', type=str, default='coco_roc')
	parser = parser.parse_args(args)

	cls_name = classes[parser.cls_id]
	save_name = os.path.join(parser.saves_dir,cls_name+'_gt_labels.npy')
	gt_labels = np.load(save_name)
	preds_list = []
	for version in parser.versions:
		save_name = os.path.join(parser.saves_dir, cls_name+'_model{:1.1f}_preds.npy'.format(version))
		preds_list.append(np.load(save_name))
	num_curves = len(preds_list)
	plt.title('ROC curve')
	plt.plot([0,1],[0,1],'r--')
	for curve_idx in range(num_curves):
		fpr, tpr, _ = metrics.roc_curve(gt_labels, preds_list[curve_idx])
		roc_auc = metrics.auc(fpr, tpr)
		plt.plot(fpr, tpr, label='model{:1.1f},AUC={:1.2f}'.format(parser.versions[curve_idx], roc_auc))
	plt.legend(loc='lower right')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.ylabel('Recall / True positive rate')
	plt.xlabel('False alarm rate / False positive rate')
	plt.show()


def get_preds(args=None):
	parser = argparse.ArgumentParser(description='Simple testing script.')
	parser.add_argument('--cls_id', help='class id', type=int)
	parser.add_argument('--version', help='model version', type=float)
	parser.add_argument('--resume_epoch', help='trained model for resume', type=int)
	parser.add_argument('--set_name', help='imply attack goal', type=str, default='test_digi_ifgsm_hiding')
	parser.add_argument('--gamma', help='gamma for the SoftL1Loss', type=float, default=9.0)
	parser.add_argument('--checkpoints', help='checkpoints path', type=str, default='voc_checkpoints')
	parser.add_argument('--saves_dir', help='the save path for tested reconstruction error', type=str, default='voc_reconstruction_error')
	parser.add_argument('--batch_size', help='batch size for optimization', type=int, default=1)
	parser = parser.parse_args(args)

	batch_size = parser.batch_size
	if not os.path.isdir(parser.saves_dir):
		os.mkdir(parser.saves_dir)

	cls_name = classes[parser.cls_id]
	parser.checkpoints = '_'.join([parser.checkpoints,cls_name])
	
	checkpoint_name = os.path.join(parser.checkpoints, 'model_{:1.1f}_epoch{:d}.pt'.format(
		parser.version, parser.resume_epoch))
	if not os.path.isfile(checkpoint_name):
		raise ValueError('No checkpoint file {:s}'.format(checkpoint_name))
	assert batch_size==1

	print('[data prepare]....')
	cls_dir = "../context_profile/voc_detection_{:s}_p10/"\
		.format(cls_name)
	dataloader_test = DataLoader(Fetch(parser.set_name, root_dir=cls_dir), batch_size=batch_size, num_workers=1, shuffle=False)

	print('[model prepare]....')
	use_gpu = torch.cuda.device_count()>0
	model = AutoEncoder(parser.gamma)
	if use_gpu:
		model = torch.nn.DataParallel(model).cuda()
	model.load_state_dict(torch.load(checkpoint_name))
	print('model loaded from {:s}'.format(checkpoint_name))
	
	print('[model testing]...')
	model.eval()
	preds = []
	with torch.no_grad():
		for sample in iter(dataloader_test):
			if use_gpu:
				data = sample['data'].cuda().float()
				
			else:
				data = sample['data'].float()
			loss = model(data)
			preds.append(float(loss))
	preds_name = '_model{:1.1f}_'+parser.set_name
	save_name = os.path.join(parser.saves_dir, cls_name+preds_name.format(parser.version))
	np.save(save_name, preds)
	print('save preds in {:s}'.format(save_name))
	
	
if __name__ == '__main__':
	get_preds()
