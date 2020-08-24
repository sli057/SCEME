import torch 
import torch.nn as nn
import numpy as np 
class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args 
	def forward(self, x):
		return x.view(self.shape)

class SmoothL1Loss(nn.Module):
	def __init__(self, gamma=9.0):
		self.gamma = gamma
		super(SmoothL1Loss, self).__init__()
	def forward(self, pred, gt):
		diff = torch.abs(gt-pred)
		loss = torch.where(
			torch.le(diff, self.gamma),
			0.5 * self.gamma * torch.pow(diff,2),
			diff - 0.5/self.gamma
			)
		#loss = torch.mean(loss, [0,1,3])
		#weights = torch.tensor([4,1,1,1,1]).cuda()
		#loss = torch.mul(loss, weights)
		return loss.mean().view(-1)

class AutoEncoder(nn.Module):
	def __init__(self, gamma=9.0):
		super(AutoEncoder, self).__init__()		
		
		self.loss_func = SmoothL1Loss(gamma)
		#self.state_dict()
		#[bs, 1, 5, 4096]
		self.compress1 = nn.Linear(in_features=4096, out_features=256)
		self.non_linear_c1 = nn.LeakyReLU()
		#[bs, 1, 5, 256]
		self.relation1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5,1), padding=(2,0))
		self.non_linear_r1 = nn.LeakyReLU()
		#[bs, 64, 5, 256]
		self.relation2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(5,1))
		self.non_linear_r2 = nn.LeakyReLU()
		#[bs, 16, 1, 256]
		#self.compress2 = nn.Conv2d(in_channels=16, out_channels=16*16, kernel_size=(1,256), groups=16)
		self.compress2 = nn.Linear(in_features=256, out_features=16)
		self.non_linear_c2 = nn.LeakyReLU()
		#[bs, 16*16, 1, 1]
		self.reshape_1 = Reshape(-1, 16*16)
		self.bottle_neck = nn.Linear(in_features=16*16, out_features=7)
		#bottle_neck = nn.Linear(in_features=16*16, out_features=3)
		self.reverse_bottle_neck = nn.Linear(in_features=7, out_features=16*16)
		self.reverse_reshape = Reshape(-1, 16, 1, 16)
		#self.reverse_reshape = Reshape(-1, 16*16, 1, 1)
		self.reverse_nonlinear_c2 = nn.LeakyReLU()
		#self.reverse_compress2 = nn.ConvTranspose2d(in_channels=16*16, out_channels=16, kernel_size=(1,256), groups=16)
		self.reverse_compress2 = nn.Linear(in_features=16, out_features=256)
		self.reverse_nonlinear_r2 = nn.LeakyReLU()
		self.reverse_relation2 = nn.ConvTranspose2d(in_channels=16, out_channels=64, kernel_size=(5,1))
		self.reverse_nonlinear_r1 = nn.LeakyReLU()
		self.reverse_relation1 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5,1), padding=(2,0))
		self.reverse_non_linear_c1 = nn.LeakyReLU()
		self.reverse_compress1 = nn.Linear(in_features=256, out_features=4096)
		self.sigmoid = 	nn.Sigmoid()

	def forward(self, x1, x2=None):
		orig_x1 = x1
		#print(x1.size())
		x1 = self.non_linear_c1(self.compress1(x1))
		#print(x1.size())
		x1 = self.non_linear_r1(self.relation1(x1))
		#print(x1.size())
		x1 = self.non_linear_r2(self.relation2(x1))
		#print(x1.size())
		x1 = self.non_linear_c2(self.compress2(x1))
		#print(x1.size())
		x1 = self.bottle_neck(self.reshape_1(x1))
		#print(x1.size())
		x1 = self.reverse_reshape(self.reverse_bottle_neck(x1))
		#print(x1.size())
		x1 = self.reverse_compress2(self.reverse_nonlinear_c2(x1))
		#print(x1.size())
		x1 = self.reverse_relation2(self.reverse_nonlinear_r2(x1))
		#print(x1.size())
		x1 = self.reverse_relation1(self.reverse_nonlinear_r1(x1))
		#print(x1.size())
		x1 = self.reverse_compress1(self.reverse_non_linear_c1(x1))
		#print(x1.size())
		x1 = self.sigmoid(x1)
		#print(x1.size())
		loss = self.loss_func(x1, orig_x1)
		return loss 


if __name__ == '__main__':
	x1 = np.ones([2,1,5,4096])
	x2 = np.ones([2,1,1,256])
	x1 = torch.tensor(x1, dtype=torch.float32)
	x2 = torch.tensor(x2, dtype=torch.float32)
	model = AutoEncoder()
	print(model.forward(x1,x2))



