from test_AE_aux import pred_box_trans
from get_data_per_cat import compare_iou
import tensorflow as tf 
import numpy as np
from attack_aux import build_test_graph

sess, net, fetch_list = build_test_graph(net_name="VGGnet_wt_context")

def is_valid(im_cv, im, im_info, one_box, t_id, iou_thred=0.7):
	# one_box [x1,y1, x2, y2, cls_id]
	gt_box, f_id = one_box[:4], int(one_box[-1])
	assert t_id is not None
	feed_dict = {net.data: np.expand_dims(im, axis=0),
				net.im_info: np.expand_dims(im_info, axis=0)}
	cls_prob, box_deltas, rois = sess.run(fetch_list, feed_dict=feed_dict)
	pred_boxes = pred_box_trans(rois, cls_prob, box_deltas, im_info[-1], im_cv.shape)
	scores = cls_prob #[num_box, num_class]
	node_overlaps = compare_iou(pred_boxes, gt_box)
	idx_list = np.where(node_overlaps>=iou_thred)[0]
	if len(idx_list) == 0:
		return False 
	pred_cls_ids = np.argmax(scores, axis=1)[idx_list] # predictions of the related bboxes
	pred_f = np.where(pred_cls_ids==f_id)[0]
	pred_t = np.where(pred_cls_ids==t_id)[0]
	if f_id == t_id:
		return len(pred_t)>0
	return len(pred_f)==0 and len(pred_t)>0

	


