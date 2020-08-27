import torch
from torch.utils.data import DataLoader
from fetch import Fetch
from auto_encoder import AutoEncoder
import numpy as np 
import argparse
import os 
from config import voc_classes

classes = voc_classes


def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script.')
	parser.add_argument('--cls_id', help='class id', type=int)
	parser.add_argument('--version', help='model version', type=float)
	parser.add_argument('--gamma', help='gamma for the SoftL1Loss', type=float, default=9.0)
	parser.add_argument('--lr', help='lr for optimization', type=float, default=1e-4)
	parser.add_argument('--epoches', help='num of epoches for optimization', type=int, default=4)
	parser.add_argument('--resume_epoch', help='trained model for resume', type=int, default=0)
	parser.add_argument('--batch_size', help='batch size for optimization', type=int, default=10)
	parser.add_argument('--checkpoints', help='checkpoints path', type=str, default='voc_checkpoints')
	parser = parser.parse_args(args)

	cls_name = classes[parser.cls_id]
	parser.checkpoints = '_'.join([parser.checkpoints,cls_name])
	if not os.path.isdir(parser.checkpoints):
		os.mkdir(parser.checkpoints)
	print('will save checkpoints in '+parser.checkpoints)
	cls_dir = "../context_profile/voc_detection_{:s}_p10/"\
		.format(cls_name)
	batch_size = parser.batch_size
	print('[data prepare]....')
	dataloader_train = DataLoader(Fetch('train_benign', root_dir=cls_dir), batch_size=batch_size, num_workers=2, shuffle=True)

	print('[model prepare]....')
	use_gpu = torch.cuda.device_count()>0

	model = AutoEncoder(parser.gamma)
	if use_gpu:
		model = torch.nn.DataParallel(model).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=parser.lr)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
	if parser.resume_epoch > 0 :
		checkpoint_name = os.path.join(parser.checkpoints, 'model_{:1.1f}_epoch{:d}.pt'.format(parser.version, parser.resume_epoch))
		if not os.path.isfile(checkpoint_name):
			raise ValueError('No checkpoint file {:s}'.format(checkpoint_name))
		model.load_state_dict(torch.load(checkpoint_name))
		print('model loaded from {:s}'.format(checkpoint_name))

	print('[model training]...')
	loss_hist = []
	epoch_loss = []
	num_iter = len(dataloader_train)
	for epoch_num in range(parser.resume_epoch, parser.epoches):
		model.train()
		for iter_num, sample in enumerate(dataloader_train):
			if True:#try:
				optimizer.zero_grad()
				if use_gpu:
					data = sample['data'].cuda().float()
				else:
					data = sample['data'].float()
					
				loss = model(data).mean()
				if bool(loss==0):
					continue 
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
				optimizer.step()
				epoch_loss.append(float(loss))
				loss_hist.append(float(loss))
				if iter_num % 30 == 0:
					print('Epoch {:d}/{:d} | Iteration: {:d}/{:d} | loss: {:1.5f}'.format(
						epoch_num+1, parser.epoches, iter_num+1, num_iter, float(loss)))
				if iter_num % 3000 == 0:
					scheduler.step(np.mean(epoch_loss))
					epoch_loss = []
		if epoch_num < 1:
			continue
		checkpoint_name = os.path.join(parser.checkpoints, 'model_{:1.1f}_epoch{:d}.pt'.format(parser.version, epoch_num+1))
		torch.save(model.state_dict(), checkpoint_name)
		print('Model saved as {:s}'.format(checkpoint_name))

	np.save('loss_hist.npy', loss_hist)


if __name__ == '__main__':
	main()