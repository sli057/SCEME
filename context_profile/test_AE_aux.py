import numpy as np
import matplotlib.pyplot as plt
import cv2
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv


def pred_box_trans(rois, cls_pred, bbox_deltas, im_scale, im_shape):
	"""
	input:
		rois: output get from RPN, [256, 5],[prob, x1, x2, y1, y2] for top 256 proposals
		cls_pred: output get from the detection net, [256, 21] for top 256 proposals
		bbox_deltas: output get from the detection net, [256, 21*4] for top 256 proposals
		im_scale: 
	output:
		Given cls_pred, get the exact pred bboxes on scaled im
	"""
	boxes = rois[:,1:5]/ im_scale # prob first, then bbox
	boxes = bbox_transform_inv(boxes, bbox_deltas)
	boxes = clip_boxes(boxes, im_shape) # [num_box]
	cat_ids = np.argmax(cls_pred, axis=1)
	pred_boxes = np.zeros([0,4])
	for box_id, cat_id in enumerate(cat_ids):
		pred_boxes = np.vstack((pred_boxes, 
			boxes[box_id,cat_id*4:(cat_id+1)*4]))
	pred_boxes *= im_scale
	return pred_boxes



def get_image_prepared(cfg,roidb,target_size=None,perturbations=None,zero_mean=True):
	if target_size is None:
		assert len(cfg.TEST.SCALES) == 1
		target_size = cfg.TEST.SCALES[0]
	#print("load image: "+roidb['image'])
	im_cv = cv2.imread(roidb['image'])
	im = im_cv.astype(np.float32, copy=True)
	
	im_size_min = np.min(im.shape[0:2])
	im_size_max = np.max(im.shape[0:2])
	im_scale = min([float(target_size) / im_size_min,
					float(cfg.TEST.MAX_SIZE) / im_size_max])
	# the biggest axis should not be more than MAX_SIZE
	im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
				interpolation=cv2.INTER_LINEAR)
	if perturbations is not None:
		im = np.clip(im+perturbations,0,255)
	if zero_mean:
		im -= cfg.PIXEL_MEANS
	im_info = np.array([im.shape[0],im.shape[1],im_scale],
					dtype=np.float32)
	# exactly same as data_layer.forward()
	gt_inds = np.where(roidb['gt_classes']!=0)[0]
	gt_boxes = np.empty((len(gt_inds),5), dtype=np.float32)
	gt_boxes[:,0:4] = roidb['boxes'][gt_inds,:]*im_scale
	gt_boxes[:,4] = roidb['gt_classes'][gt_inds]
	return im_cv, im, im_info, gt_boxes

def vis_detections(class_name, dets, im_scale, title, thresh=0.8):
	"""Visual debugging of detections."""
	for i in xrange(np.minimum(10, dets.shape[0])):
		bbox = dets[i, :4] * im_scale
		score = dets[i, -1]
		if score > thresh:
			plt.gca().add_patch(
				plt.Rectangle((bbox[0], bbox[1]),
							  bbox[2] - bbox[0],
							  bbox[3] - bbox[1], fill=False,
							  edgecolor='g', linewidth=3)
				)
			plt.gca().text(bbox[0], bbox[1] - 2,
				 '{:s} {:.3f}'.format(class_name, score),
				 bbox=dict(facecolor='blue', alpha=0.5),
				 fontsize=14, color='white')

			plt.title(title)#('{}  {:.3f}'.format(class_name, score))
	return
def vis_detections_bbox(class_names, dets, title):
	"""Visual debugging of detections."""
	for i in xrange(np.minimum(10, dets.shape[0])):
		bbox = dets[i, :4] 
		cls = int(dets[i, -1])
		if True:
			plt.gca().add_patch(
				plt.Rectangle((bbox[0], bbox[1]),
							  bbox[2] - bbox[0],
							  bbox[3] - bbox[1], fill=False,
							  edgecolor='g', linewidth=3)
				)
			plt.gca().text(bbox[0], bbox[1] - 2,
				 '{:s}'.format(class_names[cls]),
				 bbox=dict(facecolor='blue', alpha=0.5),
				 fontsize=14, color='white')

			plt.title(title)#('{}  {:.3f}'.format(class_name, score))
	return