import numpy as np 
import pickle 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def create_mask(shape, box, content=None):
	mask = np.zeros([int(shape[0]), int(shape[1]), 3])
	x1, y1, x2, y2 = np.array(box, dtype=np.int)
	if content is not None:
		mask[y1:y2,x1:x2] = content
	else:
		mask[y1:y2,x1:x2] = 1
	return mask 


def create_sticker(shape, box):
	sticker = np.zeros([int(shape[0]), int(shape[1]), 3])
	x1, y1, x2, y2 = np.array(box, dtype=np.int)
	n_x1 = x1 + 0.25*(x2-x1)
	n_y1 = y1 + 0.25*(y2-y1)
	n_x2 = x2 - 0.25*(x2-x1)
	n_y2 = y2 - 0.25*(y2-y1)
	x_seg = 1.0/7.0*(n_x2-n_x1)
	y_seg = 1.0/4.0*(n_y2-n_y1)
	box = dict()
	box[1] = (n_x1, 			n_y1, 			n_x1+x_seg, 	n_y1+y_seg	)
	box[2] = (n_x1, 			n_y1+2*y_seg,	n_x1+x_seg, 	n_y1+3*y_seg)
	box[3] = (n_x1+2*x_seg,		n_y1+y_seg,		n_x1+3*x_seg,	n_y1+2*y_seg)
	box[4] = (n_x1+2*x_seg,		n_y1+3*y_seg,	n_x1+3*x_seg,	n_y1+4*y_seg)
	box[5] = (n_x1+4*x_seg,		n_y1,			n_x1+5*x_seg,	n_y1+y_seg)
	box[6] = (n_x1+4*x_seg,		n_y1+2*y_seg,	n_x1+5*x_seg,	n_y1+3*y_seg)
	box[7] = (n_x1+6*x_seg,		n_y1+y_seg,		n_x1+7*x_seg,	n_y1+2*y_seg)
	box[8] = (n_x1+6*x_seg,		n_y1+3*y_seg,	n_x1+7*x_seg,	n_y1+4*y_seg)
	for id in range(1,9):
		xx1, yy1, xx2, yy2 = np.array(box[id], dtype=np.int)
		sticker[yy1:yy2,xx1:xx2] = 1 
	return sticker


class save_structure:
	def __init__(self, dense_matrix, im_shape, box_shape):
		self.dense = np.int32(dense_matrix)
		self.im_shape = np.int32(im_shape)
		self.box_shape = np.int32(box_shape)


class Block_Matrix:
	def save(self, name, matrix, im_shape, box_shape):
		b_x1, b_y1, b_x2, b_y2 = np.array(box_shape, dtype=np.int)
		if len(np.shape(matrix)) != 3:
			raise ValueError("implementation for 3-d matrix only")		
		data = save_structure(matrix[b_y1:b_y2,b_x1:b_x2,:], im_shape, box_shape)
		pickle.dump(data, open(name+'.p', 'w'))

	def load(self, name, display=False):
		if '.p' in name:
			data = pickle.load(open(name, 'r'))
		else:
			data = pickle.load(open(name+'.p', 'r'))
		matrix = create_mask(data.im_shape, data.box_shape,content=data.dense)
		return matrix

block_matrix = Block_Matrix()
if __name__ == "__main__":
	"""
	im_shape = (100,100)
	box_shape = (20, 20, 80, 80)
	sticker = create_sticker(im_shape, box_shape)
	plt.plot(2,1,1)
	plt.axis('off')
	plt.imshow(sticker)
	plt.show()
	raise SystemExit("Finished")
	"""

	H, W = 100, 100
	b_x1, b_y1 = 20, 80
	b_x2, b_y2 = 40, 100
	im_shape = (H,W)
	box_shape = (b_x1, b_y1, b_x2, b_y2)
	matrix = np.zeros((H,W,3))
	matrix[b_y1:b_y2, b_x1:b_x2,:] = 1
	#plt.subplot(2,1,1)
	#plt.axis('off')
	#plt.imshow(matrix)
	name = 'tmp'
	block_matrix.save(name, matrix, im_shape, box_shape)
	matrix = block_matrix.load(name)
	#plt.subplot(2,1,2)
	#plt.axis('off')
	#plt.imshow(matrix)
	#plt.show()







