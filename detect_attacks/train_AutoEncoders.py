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
	parser.add_argument('--cls_id', help='cls name', type=int)
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
	dataloader_train = DataLoader(Fetch('train', root_dir=cls_dir), batch_size=batch_size, num_workers=2, shuffle=True)
	

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
	# tensorboard
	max_value = 0
	loss_hist = []
	epoch_loss = []
	num_iter = len(dataloader_train)
	for epoch_num in range(parser.resume_epoch, parser.epoches):
		model.train()
		for iter_num, sample in enumerate(dataloader_train):
			if iter_num > 6000:
				break
			if True:#try:
				optimizer.zero_grad()
				if use_gpu:
					data1 = sample['data1'].cuda().float()
					#data2 = sample['data2'].cuda().float()
				else:
					data1 = sample['data1'].float()
					#data2 = sample['data2'].float()
				#loss = model(data1,data2)
				#[bs,1, 5, 4096]
				max_value = max(max_value, np.max(sample['data1'][:,:,0,:].numpy()))
				loss = model(data1).mean()
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

				if False:#iter_num % 3000 == 0:
					
					model.eval()
					with torch.no_grad():
						## eval for the pos val
						val_pos_loss = []
						for sample in iter(dataloader_val_pos):
							if use_gpu:
								data1 = sample['data1'].cuda().float()
								#data2 = sample['data2'].cuda().float()
							else:
								data1 = sample['data1'].float()
								#data2 = sample['data2'].float()
							#loss = model(data1,data2)
							loss = model(data1).mean()
							val_pos_loss.append(float(loss))
						print('Eval for val_pos: mean {:1.5f}, std {:1.5f}'.format(np.mean(val_pos_loss), np.std(val_pos_loss)))
						val_neg_loss = []
						for sample in iter(dataloader_val_neg):
							if use_gpu:
								data1 = sample['data1'].cuda().float()
								#data2 = sample['data2'].cuda().float()
							else:
								data1 = sample['data1'].float()
								#data2 = sample['data2'].float()
							#loss = model(data1,data2)
							loss = model(data1).mean()
							val_neg_loss.append(float(loss))
						print('Eval for val_neg: mean {:1.5f}, std {:1.5f}'.format(np.mean(val_neg_loss), np.std(val_neg_loss)))
					model.train()
			"""
			except Exception as e:
				print(e)
				continue 
			"""
		if epoch_num < 1:
			continue
		checkpoint_name = os.path.join(parser.checkpoints, 'model_{:1.1f}_epoch{:d}.pt'.format(parser.version, epoch_num+1))
		torch.save(model.state_dict(), checkpoint_name)
		print('Model saved as {:s}'.format(checkpoint_name))
		print(max_value)
		
		

	np.save('loss_hist.npy', loss_hist)

	print('[model testing]...')

if __name__ == '__main__':
	main()