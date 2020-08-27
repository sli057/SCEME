import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random


class Stickers:
	def __init__(self, num_sticker, shape):
		if num_sticker == 8:
			self.locate_stickers = self.eight_stickers
		elif num_sticker == 5:
			self.locate_stickers = self.five_stickers
		elif num_sticker == 3:
			self.locate_stickers = self.three_stickers
		elif num_sticker == 2:
			self.locate_stickers = self.two_stickers
		else:
			print('Valid number of stickers are: 2, 3, 5, 8.')
			raise ValueError("{:d} is not a valid number".format(num_sticker))

		if shape == 'rectangular':
			self.shape_sticker = self.rectangular_sticker
		elif shape == 'circular':
			# right now only support big circular/triangle
			assert num_sticker == 2
			self.shape_sticker = self.circular_sticker
		elif shape == 'triangular':
			assert num_sticker == 2
			self.shape_sticker = self.triangular_sticker
		else:
			print('Valid shapes are: rectangular, circular, and  triangular.')
			raise ValueError("{:s} is not a valid shape".format(shape))

	def eight_stickers(self, n_w, n_h):
		x_seg = 1.0/7.0*n_w
		y_seg = 1.0/4.0*n_h
		box = dict()
		box[1] = (0, 			0, 			0+x_seg, 	0+y_seg	)
		box[2] = (0, 			0+2*y_seg,	0+x_seg, 	0+3*y_seg)
		box[3] = (0+2*x_seg,	0+y_seg,	0+3*x_seg,	0+2*y_seg)
		box[4] = (0+2*x_seg,	0+3*y_seg,	0+3*x_seg,	0+4*y_seg)
		box[5] = (0+4*x_seg,	0,			0+5*x_seg,	0+y_seg)
		box[6] = (0+4*x_seg,	0+2*y_seg,	0+5*x_seg,	0+3*y_seg)
		box[7] = (0+6*x_seg,	0+y_seg,	0+7*x_seg,	0+2*y_seg)
		box[8] = (0+6*x_seg,	0+3*y_seg,	0+7*x_seg,	0+4*y_seg)
		return box

	def five_stickers(self, n_w, n_h):
		x_seg = 1.0/5.0*n_w
		y_seg = 1.0/3.0*n_h
		box = dict()
		box[1] = (0, 			0, 			0+x_seg, 	0+y_seg	)
		box[2] = (0, 			0+2*y_seg,	0+x_seg, 	0+3*y_seg)
		box[3] = (0+2*x_seg,	0+1*y_seg,	0+3*x_seg,	0+2*y_seg)
		box[4] = (0+4*x_seg,	0,			0+5*x_seg,	0+y_seg)
		box[5] = (0+4*x_seg,	0+2*y_seg,	0+5*x_seg,	0+3*y_seg)
		return box

	def three_stickers(self, n_w, n_h):
		x_seg = 1.0/5.0*n_w
		y_seg = 1.0/3.0*n_h
		box = dict()
		box[1] = (0, 			0+y_seg, 	0+x_seg, 	0+2*y_seg	)
		box[2] = (0+2*x_seg,	0+y_seg,	0+3*x_seg,	0+2*y_seg)
		box[3] = (0+4*x_seg,	0+y_seg,	0+5*x_seg,	0+3*y_seg)
		return box

	def two_stickers(self, n_w, n_h):
		x_seg = 1.0/3.0*n_w
		y_seg = 1.0/1.0*n_h
		box = dict()
		box[1] = (0, 			0, 	0+x_seg, 	0+y_seg	)
		box[3] = (0+2*x_seg,	0,	0+3*x_seg,	0+y_seg)
		return box


	def rectangular_sticker(self, w, h):
		return 1 

	def circular_sticker(self, w, h, center=None, radius=None):
		if center is None: # use the middle of the image
			center = (int(w/2), int(h/2))
		if radius is None: # use the smallest distance between the center and image walls
			radius = min(center[0], center[1], w-center[0], h-center[1])
		Y, X = np.ogrid[:h, :w]
		dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
		mask = dist_from_center <= radius
		return mask

	def triangular_sticker(self, w, h):
		vertices = [(0, random.randint(0, h-1)), (random.randint(0, w-1), 0), (w,h) ]
		img = Image.new('L', (w,h), 0)
		ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)
		mask = np.array(img)
		return mask

	def create_stickers(self, shape, box):
		stickers = np.zeros([int(shape[0]), int(shape[1])])
		# only center 1/4 area for perturbation
		x1, y1, x2, y2 = np.array(box, dtype=np.int)
		n_x1 = int(x1 + 0.25*(x2-x1))
		n_y1 = int(y1 + 0.25*(y2-y1))
		n_x2 = int(x2 - 0.25*(x2-x1))
		n_y2 = int(y2 - 0.25*(y2-y1))
		n_w, n_h = n_x2-n_x1, n_y2-n_y1

		sticker_locations = self.locate_stickers(n_w, n_h)
		
		for loc_idx in sticker_locations:
			# for each location, add the sticker 
			s_x1, s_y1, s_x2, s_y2 = np.array(sticker_locations[loc_idx],dtype=np.int)
			s_w, s_h = s_x2-s_x1, s_y2-s_y1
			sticker = self.shape_sticker(s_w, s_h)
			# add offset
			xx1, yy1, xx2, yy2 = n_x1+s_x1, n_y1+s_y1, n_x1+s_x2, n_y1+s_y2
			stickers[yy1:yy2,xx1:xx2] = sticker
		stickers = np.stack([stickers, stickers, stickers], axis=-1)
		return stickers 

if __name__ == '__main__':
	
	shape = [200, 400, 3]
	box = [100,100, 350,150]
	
	
	
	plt.subplot(2,2,1)
	image = np.zeros(shape)
	sticker_instance = Stickers(8,"rectangular")
	sticker = sticker_instance.create_stickers(shape, box)
	plt.axis('off')
	plt.imshow(sticker)
	plt.title('8, rectangular')

	plt.subplot(2,2,2)
	image = np.zeros(shape)
	sticker_instance = Stickers(2,"circular")
	sticker = sticker_instance.create_stickers(shape, box)
	plt.axis('off')
	plt.imshow(sticker)
	plt.title('2, circular')


	plt.subplot(2,2,3)
	image = np.zeros(shape)
	sticker_instance = Stickers(3,"rectangular")
	sticker = sticker_instance.create_stickers(shape, box)
	plt.axis('off')
	plt.imshow(sticker)
	plt.title('3, rectangular')
	
	
	plt.subplot(2,2,4)
	image = np.zeros(shape)
	sticker_instance = Stickers(2,"triangular")
	sticker = sticker_instance.create_stickers(shape, box)
	plt.axis('off')
	plt.imshow(sticker)
	plt.title('2, triangular')

	plt.show()
	


		 
