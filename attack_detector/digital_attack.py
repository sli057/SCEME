from __future__ import print_function
import sys, os, argparse
import numpy as np

sys.path.append('../lib')
sys.path.append('../context_profile')
from fast_rcnn.config import cfg, cfg_from_file
from utils.timer import Timer
from test_AE_aux import get_image_prepared
from valid_detection_wt_context import is_valid as is_valid_wt_context
from valid_detection_wo_context import is_valid as is_valid_wo_context
from block_matrix import block_matrix, create_mask
from attack_aux import build_digital_adv_graph, prepare_dataset
from appear_aux import generate_appear_box


def get_p_box_IFGSM(net, im_cv, im, im_info, gt_boxes, original_id, box_idx, mask, sess, grad, max_iteration=10):
    p_im = np.copy(im)
    iteration = 0
    while iteration < max_iteration:
        iteration += 1
        feed_dict = {net.data: np.expand_dims(p_im, axis=0),
                     net.im_info: np.expand_dims(im_info, axis=0),
                     net.gt_boxes: gt_boxes,
                     net.keep_prob: 1.0}
        cur_grad = sess.run(grad, feed_dict=feed_dict)
        p = np.multiply(np.squeeze(cur_grad), mask)
        p_im = np.clip(p_im + p + cfg.PIXEL_MEANS, 0, 255) - cfg.PIXEL_MEANS
    if is_valid_wt_context(im_cv, p_im, im_info, gt_boxes[box_idx], original_id):  # iteration < max_iteration:
        return np.multiply(p_im - im, mask)
    return None


def get_p_box_FGSM(net, im_cv, im, im_info, gt_boxes, original_id, box_idx, mask, sess, grad, max_iteration=10):
    p_im = np.copy(im)
    iteration = 0
    if True:  # while iteration < max_iteration:
        iteration += 1
        feed_dict = {net.data: np.expand_dims(p_im, axis=0),
                     net.im_info: np.expand_dims(im_info, axis=0),
                     net.gt_boxes: gt_boxes,
                     net.keep_prob: 1.0}
        cur_grad = sess.run(grad, feed_dict=feed_dict)
        p = np.multiply(np.squeeze(cur_grad), mask) * 10
        p_im = np.clip(p_im + p + cfg.PIXEL_MEANS, 0, 255) - cfg.PIXEL_MEANS
    if is_valid_wt_context(im_cv, p_im, im_info, gt_boxes[box_idx], original_id):  # iteration < max_iteration:
        return np.multiply(p_im - im, mask)
    return None


def get_p_set(im_set, im_list, save_dir, attack_type='miscls', net_name="VGGnet_wo_context"):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if attack_type == 'appear':
        perturbation_name = '/'.join([save_dir, 'im{:d}_box({:1.0f}-{:1.0f}-{:1.0f}-{:1.0f})_iou{:1.3f}_t{:d}'])
    else:
        perturbation_name = '/'.join([save_dir, 'im{:d}_box{:d}_f{:d}_t{:d}'])

    # some configuration
    extra_config = "../experiments/cfgs/faster_rcnn_end2end.yml"
    cfg_from_file(extra_config)

    # prepare data
    imdb = prepare_dataset(im_set, cfg)
    # prepare graph and sess
    sess, net, grad = build_digital_adv_graph(net_name, im_set)

    num_images = len(im_list)
    _t = Timer()
    for idx, i in enumerate(im_list):
        im_cv, im, im_info, gt_boxes = get_image_prepared(cfg, imdb.roidb[idx])
        num_gt_boxes = len(gt_boxes)
        _t.tic()
        for box_id in range(num_gt_boxes):
            valid = is_valid_wo_context(im_cv, im, im_info, gt_boxes[box_id],
                                        f_id=int(gt_boxes[box_id][-1]))  # changed here
            if not valid:
                break
        if not valid:
            print("ignore the image since at least one object is not detected correctly")
            continue
        if attack_type == 'appear':
            ori_gt_boxes = gt_boxes
            new_gt_boxes, iou_list = generate_appear_box(im_info, gt_boxes)  # (x1, y1, x2, y2, gt_cls=0)
            num_iter = len(new_gt_boxes)
        else:
            num_iter = num_gt_boxes

        for box_id in range(num_iter):
            new_box_id = box_id
            if attack_type == 'appear':
                box_id = -1
                gt_boxes = np.concatenate([ori_gt_boxes,
                                           np.expand_dims(new_gt_boxes[new_box_id], axis=0)])
            gt_cls = int(gt_boxes[box_id, -1])
            gt_cls_name = imdb._classes[gt_cls]
            for target_cls, target_cls_name in enumerate(imdb._classes):
                if attack_type == 'hiding' and target_cls != 0:
                    continue
                elif target_cls == int(gt_boxes[box_id, -1]):
                    continue
                elif attack_type != 'hiding' and target_cls == 0:
                    continue
                mask = create_mask(im_info[:2], gt_boxes[box_id, :4])
                original_id = gt_boxes[box_id, -1]
                gt_boxes[box_id, -1] = target_cls
                p = get_p_box_IFGSM(net, im_cv, im, im_info, gt_boxes, original_id, box_id, mask, sess, grad)
                # p = get_p_box_FGSM(net,im_cv, im, im_info, gt_boxes, original_id, box_id, mask, sess, grad)

                gt_boxes[box_id, -1] = gt_cls
                if p is not None:
                    if attack_type == 'appear':
                        save_name = perturbation_name.format(i, gt_boxes[box_id, 0], gt_boxes[box_id, 1],
                                                             gt_boxes[box_id, 2], gt_boxes[box_id, 3], iou_list[box_id],
                                                             target_cls)
                    else:
                        save_name = perturbation_name.format(i, box_id, gt_cls, target_cls)
                    p = np.int32(p)

                    block_matrix.save(save_name,
                                      p, im_info[:2], gt_boxes[box_id, :4])
                    print("{:s} --> {:s} succeed." \
                          .format(gt_cls_name, imdb._classes[target_cls]))
                else:
                    print("{:s} --> {:s} failed." \
                          .format(gt_cls_name, imdb._classes[target_cls]))
        _t.toc()
        print('perturbation_generation: {:d}/{:d} {:.3f}s' \
              .format(idx + 1, num_images, _t.average_time))


def parse_parameter(args=None):
    parser = argparse.ArgumentParser(description='Digital attack script.')
    parser.add_argument('--attack_type', choices={'appear', 'hiding', 'miscls'}, default='miscls')
    return parser.parse_args(args)


if __name__ == '__main__':
    parser = parse_parameter()
    # if dataset == 'coco':
    # 	im_set = 'coco_2014_minival'
    # 	im_list = list(open( '../data/coco/annotations/coco_2014_minival.txt','r'))
    # 	im_list = [int(idx.strip().split('_')[-1]) for idx in im_list]

    im_set = "voc_2007_test"
    im_list = list(open('../data/VOCdevkit/VOC2007/ImageSets/Main/test.txt', 'r'))
    im_list = [int(idx.strip()) for idx in im_list]
    save_dir = 'IFGSM_p_' + parser.attack_type
    get_p_set(im_set, im_list, save_dir, attack_type=parser.attack_type)
