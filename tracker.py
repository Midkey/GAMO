import argparse
import sys
import time

## for drawing package
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
import torch.optim as optim
from torch.autograd import Variable
from random import randint

# sys.path.insert(0,'./modules')
from modules.sample_generator import *
from modules.data_prov import *
from modules.model import *
from modules.bbreg import *
from options import *
from modules.img_cropper import *
from modules.roi_align.modules.roi_align import RoIAlignAvg, RoIAlignMax, RoIAlignAdaMax#, RoIAlignDenseAdaMax

from motion_model import MotionModeler
from modules.graph_match import GraphMatch

import cv2

# import aum

st0 = np.random.get_state()


def set_optimizer(model, lr_base, lr_mult=opts['lr_mult'], momentum=opts['momentum'], w_decay=opts['w_decay']):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})

    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    # optimizer = optim.SGD(param_list, lr = 1., momentum=momentum, weight_decay=w_decay)
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while (len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for iter in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()  ## model transfer into evaluation mode
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx))
            model.train()  ## model transfer into train mode

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

        if opts['visual_log']:
            print("Iter %d, Loss %.4f" % (iter, loss.data[0]))


def initial_train_with_fam(model, cur_image, target_bbox, img_crop_model, criterion, init_optimizer):
    # Draw pos/neg samples
    ishape = cur_image.shape
    pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])


    # generate fam
    im_size = (cur_image.shape[1], cur_image.shape[0])

    cx, cy = target_bbox[:2] + target_bbox[-2:] / 2
    enlarge_w, enlarge_h = 10 * target_bbox[-2:]
    # scene_box = np.array([max(cx - enlarge_w / 2, 0), max(cy - enlarge_h / 2, 0),
    #                       min(cx + enlarge_w / 2, im_size[0]), min(cy + enlarge_h / 2, im_size[1])])
    scene_box = np.array([max(cx - enlarge_w / 2, 0), max(cy - enlarge_h / 2, 0),
                          min(cx + enlarge_w / 2, im_size[0]), min(cy + enlarge_h / 2, im_size[1])])

    # enlarge_w = min(enlarge_w, scene_box[2]-scene_box[0])
    # enlarge_h = min(enlarge_h, scene_box[3]-scene_box[1])
    enlarge_w = scene_box[2]-scene_box[0]
    enlarge_h = scene_box[3]-scene_box[1]

    scene_box[2] = enlarge_w
    scene_box[3] = enlarge_h

    # cropped_target_box = np.array([target_bbox[0]-scene_box[0], target_bbox[1]-scene_box[1], target_bbox[2], target_bbox[3]])
    # crop_img = cur_image[int(scene_box[1]):int(scene_box[3]), int(scene_box[0]):int(scene_box[2]), :]
    # cv2.rectangle(crop_img, (int(cropped_target_box[0]), int(cropped_target_box[1])),
    #               (int(cropped_target_box[0]+cropped_target_box[2]), int(cropped_target_box[1]+cropped_target_box[3])), (0, 0, 255), 1)
    # cv2.imshow('aaa', crop_img)
    # cv2.waitKey(0)

    new_size = (np.array([enlarge_w, enlarge_h]) * ((opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype(
        'int64')
    model.eval()
    with torch.no_grad():
        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(scene_box, (1, 4)),
                                                                 new_size)
        cropped_image = cropped_image - 128.
        with torch.no_grad():
            feat_map_tmp = model(cropped_image, out_layer='conv3')
        feat_map = feat_map_tmp.sum(dim=1)

        feat_map_mean = torch.mean(feat_map)
        ff = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
        ff = ff.squeeze().cpu().numpy()
        ff = cv2.resize(ff, (new_size[0], new_size[1]))

        feat_map[feat_map < feat_map_mean] = feat_map_mean

        feat_map = (feat_map - feat_map.min()) / (feat_map.max() - feat_map.min())
        feat_map = feat_map.squeeze().cpu().numpy()
        feat_map = cv2.resize(feat_map, (new_size[0], new_size[1]))

    w_scale = new_size[0] / enlarge_w
    h_scale = new_size[1] / enlarge_h

    cropped_target_box = np.array([target_bbox[0]-scene_box[0], target_bbox[1]-scene_box[1], target_bbox[2], target_bbox[3]])
    cropped_target_box[::2] *= w_scale
    cropped_target_box[1::2] *= h_scale
    neg_examples = gen_samples(SampleGenerator_FAM('uniform', (new_size[0], new_size[1]), feat_map),
                               cropped_target_box, opts['n_neg_init'], (0, 0)) #  opts['overlap_neg_init'])

    # plt.imshow(feat_map)
    # plt.show()
    # draw_img = cropped_image.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8) + 128
    # for rect in neg_examples:
    #     rect = list(map(int, rect))
    #     cv2.rectangle(draw_img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 1)
    # cv2.rectangle(draw_img, (int(cropped_target_box[0]), int(cropped_target_box[1])),
    #               (int(cropped_target_box[0]+cropped_target_box[2]), int(cropped_target_box[1]+cropped_target_box[3])), (0, 0, 255), 1)
    # cv2.imshow('233', draw_img)
    # cv2.waitKey(0)

    # neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 1, 2, 1.1),
    #                            target_bbox, opts['n_neg_init'], opts['overlap_neg_init'])
    neg_examples = np.random.permutation(neg_examples)

    neg_examples[:, :2] += scene_box[:2]

    # compute padded sample
    # padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
    # padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
    # padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
    # padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
    padded_scene_box = np.reshape(np.asarray((scene_box[0], scene_box[1], scene_box[2] - scene_box[0], scene_box[3] - scene_box[1])),
                                  (1, 4))

    scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))

    model.eval()
    bidx = 0
    torch.cuda.empty_cache()
    # crop_img_size = (scene_boxes[bidx, 2:4] * ((opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype(
    #     'int64')
    # cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(scene_boxes[bidx], (1, 4)),
    #                                                          crop_img_size)
    # cropped_image = cropped_image - 128.

    feat_map = feat_map_tmp

    rel_target_bbox = np.copy(target_bbox)
    rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

    batch_num = np.zeros((pos_examples.shape[0], 1))
    cur_pos_rois = np.copy(pos_examples)
    cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0], axis=0)
    scaled_obj_size = float(opts['img_size'])
    cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                   target_bbox[2:4], opts['padding'])
    cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
    cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
    cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
    cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

    batch_num = np.zeros((neg_examples.shape[0], 1))
    cur_neg_rois = np.copy(neg_examples)
    cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0], axis=0)
    cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                   target_bbox[2:4], opts['padding'])
    cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
    cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
    cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
    cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

    feat_dim = cur_pos_feats.size(-1)


    pos_feats = cur_pos_feats
    neg_feats = cur_neg_feats


    if pos_feats.size(0) > opts['n_pos_init']:
        pos_idx = np.asarray(range(pos_feats.size(0)))
        np.random.shuffle(pos_idx)
        pos_feats = pos_feats[pos_idx[0:opts['n_pos_init']], :]
    if neg_feats.size(0) > opts['n_neg_init']:
        neg_idx = np.asarray(range(neg_feats.size(0)))
        np.random.shuffle(neg_idx)
        neg_feats = neg_feats[neg_idx[0:opts['n_neg_init']], :]

    torch.cuda.empty_cache()
    model.zero_grad()

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, 200) #  opts['maxiter_init']


    with torch.no_grad():
        feat_map_tmp2 = model(cropped_image, out_layer='conv3')
    feat_map2 = feat_map_tmp2.sum(dim=1)

    feat_map2 = (feat_map2 - feat_map2.min()) / (feat_map2.max() - feat_map2.min())
    feat_map2 = feat_map2.squeeze().cpu().numpy()
    feat_map2 = cv2.resize(feat_map2, (new_size[0], new_size[1]))
    plt.subplot(121)
    plt.imshow(ff)
    plt.subplot(122)
    plt.imshow(feat_map2)
    plt.show()

    if pos_feats.size(0) > opts['n_pos_update']:
        pos_idx = np.asarray(range(pos_feats.size(0)))
        np.random.shuffle(pos_idx)
        pos_feats_all = [pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())]
    if neg_feats.size(0) > opts['n_neg_update']:
        neg_idx = np.asarray(range(neg_feats.size(0)))
        np.random.shuffle(neg_idx)
        neg_feats_all = [neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())]
    
    return pos_feats_all, neg_feats_all, feat_dim


def initial_train(model, cur_image, target_bbox, img_crop_model, criterion, init_optimizer):
    # Draw pos/neg samples
    ishape = cur_image.shape
    target_bbox = target_bbox.copy()
    target_bbox[2] = max(4, target_bbox[2])
    target_bbox[3] = max(4, target_bbox[3])
    pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                               target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
    neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 1, 2, 1.1),
                               target_bbox, opts['n_neg_init'], opts['overlap_neg_init'])
    neg_examples = np.random.permutation(neg_examples)

    cur_bbreg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 1.5, 1.1),
                                     target_bbox, opts['n_bbreg'], opts['overlap_bbreg'], opts['scale_bbreg'])

    # compute padded sample
    padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
    padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
    padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
    padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
    padded_scene_box = np.reshape(np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)),
                                  (1, 4))

    scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
    if opts['jitter']:
        ## horizontal shift
        jittered_scene_box_horizon = np.copy(padded_scene_box)
        jittered_scene_box_horizon[0, 0] -= 4.
        jitter_scale_horizon = 1.

        ## vertical shift
        jittered_scene_box_vertical = np.copy(padded_scene_box)
        jittered_scene_box_vertical[0, 1] -= 4.
        jitter_scale_vertical = 1.

        jittered_scene_box_reduce1 = np.copy(padded_scene_box)
        jitter_scale_reduce1 = 1.1 ** (-1)

        ## vertical shift
        jittered_scene_box_enlarge1 = np.copy(padded_scene_box)
        jitter_scale_enlarge1 = 1.1 ** (1)

        ## scale reduction
        jittered_scene_box_reduce2 = np.copy(padded_scene_box)
        jitter_scale_reduce2 = 1.1 ** (-2)
        ## scale enlarge
        jittered_scene_box_enlarge2 = np.copy(padded_scene_box)
        jitter_scale_enlarge2 = 1.1 ** (2)

        scene_boxes = np.concatenate(
            [scene_boxes, jittered_scene_box_horizon, jittered_scene_box_vertical, jittered_scene_box_reduce1,
             jittered_scene_box_enlarge1, jittered_scene_box_reduce2, jittered_scene_box_enlarge2], axis=0)
        jitter_scale = [1., jitter_scale_horizon, jitter_scale_vertical, jitter_scale_reduce1, jitter_scale_enlarge1,
                        jitter_scale_reduce2, jitter_scale_enlarge2]
    else:
        jitter_scale = [1.]

    model.eval()
    for bidx in range(0, scene_boxes.shape[0]):
        crop_img_size = (scene_boxes[bidx, 2:4] * ((opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype(
            'int64') * jitter_scale[bidx]
        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(scene_boxes[bidx], (1, 4)),
                                                                 crop_img_size)
        cropped_image = cropped_image - 128.

        feat_map = model(cropped_image, out_layer='conv3')

        rel_target_bbox = np.copy(target_bbox)
        rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

        batch_num = np.zeros((pos_examples.shape[0], 1))
        cur_pos_rois = np.copy(pos_examples)
        cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0], axis=0)
        scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
        cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                       target_bbox[2:4], opts['padding'])
        cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
        cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
        cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
        cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

        batch_num = np.zeros((neg_examples.shape[0], 1))
        cur_neg_rois = np.copy(neg_examples)
        cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0], axis=0)
        cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                       target_bbox[2:4], opts['padding'])
        cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
        cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
        cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
        cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

        ## bbreg rois
        batch_num = np.zeros((cur_bbreg_examples.shape[0], 1))
        cur_bbreg_rois = np.copy(cur_bbreg_examples)
        cur_bbreg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_bbreg_rois.shape[0], axis=0)
        scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
        cur_bbreg_rois = samples2maskroi(cur_bbreg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                         target_bbox[2:4], opts['padding'])
        cur_bbreg_rois = np.concatenate((batch_num, cur_bbreg_rois), axis=1)
        cur_bbreg_rois = Variable(torch.from_numpy(cur_bbreg_rois.astype('float32'))).cuda()
        cur_bbreg_feats = model.roi_align_model(feat_map, cur_bbreg_rois)
        cur_bbreg_feats = cur_bbreg_feats.view(cur_bbreg_feats.size(0), -1).data.clone()

        feat_dim = cur_pos_feats.size(-1)

        if bidx == 0:
            pos_feats = cur_pos_feats
            neg_feats = cur_neg_feats
            ##bbreg feature
            bbreg_feats = cur_bbreg_feats
            bbreg_examples = cur_bbreg_examples
        else:
            pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
            neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)
            ##bbreg feature
            bbreg_feats = torch.cat((bbreg_feats, cur_bbreg_feats), dim=0)
            bbreg_examples = np.concatenate((bbreg_examples, cur_bbreg_examples), axis=0)

    if pos_feats.size(0) > opts['n_pos_init']:
        pos_idx = np.asarray(range(pos_feats.size(0)))
        np.random.shuffle(pos_idx)
        pos_feats = pos_feats[pos_idx[0:opts['n_pos_init']], :]
    if neg_feats.size(0) > opts['n_neg_init']:
        neg_idx = np.asarray(range(neg_feats.size(0)))
        np.random.shuffle(neg_idx)
        neg_feats = neg_feats[neg_idx[0:opts['n_neg_init']], :]

    ##bbreg
    if bbreg_feats.size(0) > opts['n_bbreg']:
        bbreg_idx = np.asarray(range(bbreg_feats.size(0)))
        np.random.shuffle(bbreg_idx)
        bbreg_feats = bbreg_feats[bbreg_idx[0:opts['n_bbreg']], :]
        bbreg_examples = bbreg_examples[bbreg_idx[0:opts['n_bbreg']], :]
        # print bbreg_examples.shape

    ## open images and crop patch from obj
    extra_obj_size = np.array((opts['img_size'], opts['img_size']))
    extra_crop_img_size = extra_obj_size * (opts['padding'] + 0.6)
    replicateNum = 0
    for iidx in range(replicateNum):
        extra_target_bbox = np.copy(target_bbox)

        extra_scene_box = np.copy(extra_target_bbox)
        extra_scene_box_center = extra_scene_box[0:2] + extra_scene_box[2:4] / 2.
        extra_scene_box_size = extra_scene_box[2:4] * (opts['padding'] + 0.6)
        extra_scene_box[0:2] = extra_scene_box_center - extra_scene_box_size / 2.
        extra_scene_box[2:4] = extra_scene_box_size

        extra_shift_offset = np.clip(2. * np.random.randn(2), -4, 4)
        cur_extra_scale = 1.1 ** np.clip(np.random.randn(1), -2, 2)

        extra_scene_box[0] += extra_shift_offset[0]
        extra_scene_box[1] += extra_shift_offset[1]
        extra_scene_box[2:4] *= cur_extra_scale[0]

        scaled_obj_size = float(opts['img_size']) / cur_extra_scale[0]

        cur_extra_cropped_image, _ = img_crop_model.crop_image(cur_image, np.reshape(extra_scene_box, (1, 4)),
                                                               extra_crop_img_size)
        cur_extra_cropped_image = cur_extra_cropped_image.detach()

        cur_extra_pos_examples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2),
                                             extra_target_bbox, opts['n_pos_init'] // replicateNum,
                                             opts['overlap_pos_init'])
        cur_extra_neg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 2, 1.1),
                                             extra_target_bbox, opts['n_neg_init'] // replicateNum // 4,
                                             opts['overlap_neg_init'])

        ##bbreg sample
        cur_extra_bbreg_examples = gen_samples(SampleGenerator('uniform', (ishape[1], ishape[0]), 0.3, 1.5, 1.1),
                                               extra_target_bbox, opts['n_bbreg'] // replicateNum // 4,
                                               opts['overlap_bbreg'], opts['scale_bbreg'])

        batch_num = iidx * np.ones((cur_extra_pos_examples.shape[0], 1))
        cur_extra_pos_rois = np.copy(cur_extra_pos_examples)
        cur_extra_pos_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                cur_extra_pos_rois.shape[0], axis=0)
        cur_extra_pos_rois = samples2maskroi(cur_extra_pos_rois, model.receptive_field,
                                             (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                             opts['padding'])
        cur_extra_pos_rois = np.concatenate((batch_num, cur_extra_pos_rois), axis=1)

        batch_num = iidx * np.ones((cur_extra_neg_examples.shape[0], 1))
        cur_extra_neg_rois = np.copy(cur_extra_neg_examples)
        cur_extra_neg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)), cur_extra_neg_rois.shape[0],
                                                axis=0)
        cur_extra_neg_rois = samples2maskroi(cur_extra_neg_rois, model.receptive_field,
                                             (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                             opts['padding'])
        cur_extra_neg_rois = np.concatenate((batch_num, cur_extra_neg_rois), axis=1)

        ## bbreg rois
        batch_num = iidx * np.ones((cur_extra_bbreg_examples.shape[0], 1))
        cur_extra_bbreg_rois = np.copy(cur_extra_bbreg_examples)
        cur_extra_bbreg_rois[:, 0:2] -= np.repeat(np.reshape(extra_scene_box[0:2], (1, 2)),
                                                  cur_extra_bbreg_rois.shape[0], axis=0)
        cur_extra_bbreg_rois = samples2maskroi(cur_extra_bbreg_rois, model.receptive_field,
                                               (scaled_obj_size, scaled_obj_size), extra_target_bbox[2:4],
                                               opts['padding'])
        cur_extra_bbreg_rois = np.concatenate((batch_num, cur_extra_bbreg_rois), axis=1)

        if iidx == 0:
            extra_cropped_image = cur_extra_cropped_image

            extra_pos_rois = np.copy(cur_extra_pos_rois)
            extra_neg_rois = np.copy(cur_extra_neg_rois)
            ##bbreg rois
            extra_bbreg_rois = np.copy(cur_extra_bbreg_rois)
            extra_bbreg_examples = np.copy(cur_extra_bbreg_examples)
        else:
            extra_cropped_image = torch.cat((extra_cropped_image, cur_extra_cropped_image), dim=0)

            extra_pos_rois = np.concatenate((extra_pos_rois, np.copy(cur_extra_pos_rois)), axis=0)
            extra_neg_rois = np.concatenate((extra_neg_rois, np.copy(cur_extra_neg_rois)), axis=0)
            ##bbreg rois
            extra_bbreg_rois = np.concatenate((extra_bbreg_rois, np.copy(cur_extra_bbreg_rois)), axis=0)
            extra_bbreg_examples = np.concatenate((extra_bbreg_examples, np.copy(cur_extra_bbreg_examples)), axis=0)
    if replicateNum != 0:
        extra_pos_rois = Variable(torch.from_numpy(extra_pos_rois.astype('float32'))).cuda()
        extra_neg_rois = Variable(torch.from_numpy(extra_neg_rois.astype('float32'))).cuda()
        ##bbreg rois
        extra_bbreg_rois = Variable(torch.from_numpy(extra_bbreg_rois.astype('float32'))).cuda()

        extra_cropped_image -= 128.

        extra_feat_maps = model(extra_cropped_image, out_layer='conv3')
        # Draw pos/neg samples
        ishape = cur_image.shape

        extra_pos_feats = model.roi_align_model(extra_feat_maps, extra_pos_rois)
        extra_pos_feats = extra_pos_feats.view(extra_pos_feats.size(0), -1).data.clone()

        extra_neg_feats = model.roi_align_model(extra_feat_maps, extra_neg_rois)
        extra_neg_feats = extra_neg_feats.view(extra_neg_feats.size(0), -1).data.clone()
        ##bbreg feat
        extra_bbreg_feats = model.roi_align_model(extra_feat_maps, extra_bbreg_rois)
        extra_bbreg_feats = extra_bbreg_feats.view(extra_bbreg_feats.size(0), -1).data.clone()

        ## concatenate extra features to original_features
        pos_feats = torch.cat((pos_feats, extra_pos_feats), dim=0)
        neg_feats = torch.cat((neg_feats, extra_neg_feats), dim=0)
        ## concatenate extra bbreg feats to original_bbreg_feats
        bbreg_feats = torch.cat((bbreg_feats, extra_bbreg_feats), dim=0)
        bbreg_examples = np.concatenate((bbreg_examples, extra_bbreg_examples), axis=0)

    torch.cuda.empty_cache()
    model.zero_grad()

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])

    ##bbreg train
    if bbreg_feats.size(0) > opts['n_bbreg']:
        bbreg_idx = np.asarray(range(bbreg_feats.size(0)))
        np.random.shuffle(bbreg_idx)
        bbreg_feats = bbreg_feats[bbreg_idx[0:opts['n_bbreg']], :]
        bbreg_examples = bbreg_examples[bbreg_idx[0:opts['n_bbreg']], :]
    bbreg = BBRegressor((ishape[1], ishape[0]))
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    if pos_feats.size(0) > opts['n_pos_update']:
        pos_idx = np.asarray(range(pos_feats.size(0)))
        np.random.shuffle(pos_idx)
        pos_feats_all = [pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())]
    if neg_feats.size(0) > opts['n_neg_update']:
        neg_idx = np.asarray(range(neg_feats.size(0)))
        np.random.shuffle(neg_idx)
        neg_feats_all = [neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())]

    return pos_feats_all, neg_feats_all, feat_dim


def run_mdnet(img_list, init_bbox, gt=None, seq='seq_name ex)Basketball', savefig_dir='', display=False, history_step=15, nms_overlap=0.3, imgs=None):
    ############################################
    ############################################
    ############################################
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    iou_result = np.zeros((len(img_list), 1))

    object_num_list = []

    result[0] = np.copy(target_bbox)
    result_bb[0] = np.copy(target_bbox)

    # execution time array
    exec_time_result = np.zeros((len(img_list), 1))

    # Init model
    model = MDNet(opts['model_path'])
    if opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
    if opts['use_gpu']:
        model = model.cuda()

    # Init image crop model
    img_crop_model = imgCropper(1.)
    if opts['use_gpu']:
        img_crop_model.gpuEnable()

    # Init criterion and optimizer
    criterion = BinaryLoss()

    tic = time.time()
    # Load first image
    if imgs is not None:
        cur_image = imgs[0]
    else:
        cur_image = Image.open(img_list[0]).convert('RGB')
    cur_image = np.asarray(cur_image)


    # init fc and collect traing example
    # model.set_learnable_params(opts['ft_layers'])
    # init_optimizer = set_optimizer(model, opts['lr_init'])
    # pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion, init_optimizer)

    # init backbone
    model.set_all_params_learnable()
    init_backbone_optimizer = set_optimizer(model, 0.001)
    pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion, init_backbone_optimizer)

    # _, _, _ = initial_train_with_fam(model, cur_image, target_bbox, img_crop_model, criterion, init_backbone_optimizer)

    model.set_learnable_params(opts['ft_layers'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    spf_total = time.time() - tic
    # spf_total = 0. # no first frame

    # Display
    savefig = savefig_dir != ''
    if display or savefig:

        draw_img = cur_image.copy()
        if gt is not None:
            pt1 = np.around([gt[0, 0], gt[0, 1]]).astype(np.int)
            pt2 = np.around([gt[0, 0] + gt[0, 2], gt[0, 1] + gt[0, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

        pt1 = np.around([result_bb[0, 0], result_bb[0, 1]]).astype(np.int)
        pt2 = np.around([result_bb[0, 0] + result_bb[0, 2], result_bb[0, 1] + result_bb[0, 3]]).astype(np.int)
        cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

        cv2.imshow('show_result', draw_img)
        cv2.waitKey(0)
            
    # Main loop
    trans_f = opts['trans_f']

    last_target_bbox = target_bbox
    # offset = 0
    motion_model = MotionModeler(histoty_step=history_step)
    graph_match = GraphMatch(nms_overlap=nms_overlap, draw_graph_flag=display)

    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        if imgs is not None:
            cur_image = imgs[i]
        else:
            cur_image = Image.open(img_list[i]).convert('RGB')
        cur_image = np.asarray(cur_image)

        # Estimate target bbox
        ishape = cur_image.shape
        # offset = target_bbox - last_target_bbox
        last_target_bbox = target_bbox
        if history_step < 1:
            samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
                              target_bbox, opts['n_samples'])
            samples_weight = None
            motion_predict_rect = None
        else:
            samples, samples_weight, motion_predict_rect = motion_model.generate_samples_fast(cur_image.shape[:-1], trans_f, 1, True, target_bbox, 200, display=display, img=cur_image.copy(), ind=i)

        # samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
        #                       target_bbox, opts['n_samples'])

        padded_x1 = (samples[:, 0] - samples[:, 2] * (opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_image.shape[1]:
            padded_scene_box[0] = cur_image.shape[1] - 1
        if padded_scene_box[1] > cur_image.shape[0]:
            padded_scene_box[1] = cur_image.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (padded_scene_box[2:4] * ((opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype(
            'int64')

        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(padded_scene_box, (1, 4)),
                                                                 crop_img_size)
        cropped_image = cropped_image - 128.
        model.eval()
        feat_map = model(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(target_bbox)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, model.receptive_field, (opts['img_size'], opts['img_size']),
                                      target_bbox[2:4], opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = model(sample_feats, in_layer='fc4')
        # sample_scores = torch.nn.functional.softmax(sample_scores)
        sample_scores = sample_scores.cpu()
        # if samples_weight is not None:
        #     sample_scores[:, 1] *= torch.from_numpy(samples_weight.astype(np.float32))

        match_rect, obj_num = graph_match.update(torch.from_numpy(samples), sample_scores[:, 1], motion_predict_rect if motion_predict_rect is not None else target_bbox,
                                        cur_image.copy() if display else None, i=i)
        
        #
        # match_rect = graph_match.update_nearest(torch.from_numpy(samples), sample_scores[:, 1], motion_predict_rect if motion_predict_rect is not None else target_bbox,
        #                                 cur_image.copy() if display else None, i=i)
        #
        if match_rect is None:
            top_scores, top_idx = sample_scores[:, 1].topk(1)
            top_idx = top_idx.data.cpu().numpy()
            target_score = top_scores.data.mean()

            success = target_score > opts['success_thr']
            # if success:
            target_bbox = samples[top_idx].mean(axis=0)
            object_num_list.append(0)
        else:
            target_bbox = match_rect[0]
            target_score = match_rect[1]
            object_num_list.append(obj_num)
            success = target_score > opts['success_thr']


        # # Expand search area at failure
        if success:
            trans_f = opts['trans_f']
        else:
            trans_f = opts['trans_f_expand']

        use_motion = False
        ## Bbox regression
        if success:
            # bbreg_feats = sample_feats[top_idx, :]
            # bbreg_samples = samples[top_idx]
            # bbreg_samples = bbreg.predict(bbreg_feats.data, bbreg_samples)
            # bbreg_bbox = bbreg_samples.mean(axis=0)
            bbreg_bbox = target_bbox# + offset
            motion_model.update(target_bbox)

        else:
            # print('target bbox:', target_bbox)
            # print('motion predict:', motion_predict_rect)
            if motion_predict_rect is not None:
                target_bbox = motion_predict_rect
                use_motion = True

            bbreg_bbox = target_bbox# + offset
        # clip bbox
        target_bbox[0] = max(0, target_bbox[0])
        target_bbox[1] = max(0, target_bbox[1])
        target_bbox[0] = min(target_bbox[0], ishape[1]-target_bbox[2])
        target_bbox[1] = min(target_bbox[1], ishape[0]-target_bbox[3])

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox
        iou_result[i] = 1.

        # Data collect
        if success:

            # Draw pos/neg samples
            pos_examples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
                opts['n_pos_update'],
                opts['overlap_pos_update'])
            neg_examples = gen_samples(
                SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
                opts['n_neg_update'],
                opts['overlap_neg_update'])

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
            padded_scene_box = np.reshape(
                np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)), (1, 4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * (
                            (opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype('int64') * jitter_scale[
                                    bidx]
                cropped_image, cur_image_var = img_crop_model.crop_image(cur_image,
                                                                         np.reshape(scene_boxes[bidx], (1, 4)),
                                                                         crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = model(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(target_bbox)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0],
                                                  axis=0)
                scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats  ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > opts['n_pos_update']:
                pos_idx = np.asarray(range(pos_feats.size(0)))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > opts['n_neg_update']:
                neg_idx = np.asarray(range(neg_feats.size(0)))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())

            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)

            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # if i < 200:
        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            # pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            pos_data = torch.cat(pos_feats_all, dim=0)
            # neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.cat(neg_feats_all, dim=0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            # pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            # neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            pos_data = torch.cat(pos_feats_all, dim=0)
            neg_data = torch.cat(neg_feats_all, dim=0) # torch.stack(, 0).view(-1, feat_dim)

            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            print('## {} ##'.format(i))
            # print(i, success.item())
            draw_img = cur_image.copy()
            all_rect_img = cur_image.copy()
            all_rect_img = cv2.resize(all_rect_img, None, fx=4, fy=4)
            result_img = all_rect_img.copy()
            if gt is not None:
                pt1 = np.around([gt[i, 0], gt[i, 1]]).astype(np.int)
                pt2 = np.around([gt[i, 0] + gt[i, 2], gt[i, 1] + gt[i, 3]]).astype(np.int)
                cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)


            pt1 = np.around([result_bb[i, 0], result_bb[i, 1]]).astype(np.int)
            pt2 = np.around([result_bb[i, 0] + result_bb[i, 2], result_bb[i, 1] + result_bb[i, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if use_motion:
                cv2.circle(draw_img, (10, 10), 5, (0, 0, 255), -1)

            # show success potential bbox
            # top_k = 10
            # show_ind = np.argsort(sample_scores[:, 1].detach().numpy())[-top_k:]
            # for ind in show_ind:
            #     bbox = samples[ind]
            #     # print(sample_scores[ind])
            #     pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
            #     pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
            #         np.int)
            #     cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

            for score, bbox in zip(sample_scores[:, 1], samples):
                if score > opts['success_thr']:
                    pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
                    pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
                        np.int)
                    cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

                pt1 = np.around([bbox[0]*4, bbox[1]*4]).astype(np.int)
                pt2 = np.around([(bbox[0] + bbox[2]) * 4, (bbox[1] + bbox[3])*4]).astype(
                    np.int)
                cv2.rectangle(all_rect_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            pt1 = np.around([result_bb[i, 0]*4, result_bb[i, 1]*4]).astype(np.int)
            pt2 = np.around([(result_bb[i, 0] + result_bb[i, 2])*4, (result_bb[i, 1] + result_bb[i, 3])*4]).astype(np.int)
            cv2.rectangle(result_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if motion_predict_rect is not None:
                cx, cy = np.around(motion_predict_rect[:2] + motion_predict_rect[2:]/2).astype(np.int)
                cv2.circle(draw_img, (cx, cy), 2, (0, 0, 255), -1)

            cv2.imshow('show_result', draw_img)
            # cv2.imwrite(os.path.join('flow_pic/raw', '{:04d}.png'.format(i)), cur_image)
            # cv2.imwrite(os.path.join('flow_pic/candidate', '{:04d}.png'.format(i)), all_rect_img)
            # cv2.imwrite(os.path.join('flow_pic/result', '{:04d}.png'.format(i)), result_img)
            if i < 500:
                cv2.waitKey(1)
            else:
                cv2.waitKey(0)

        if opts['visual_log']:
            if gt is None:
                print("Frame %d/%d, Score %.3f, Time %.3f" % \
                      (i, len(img_list), target_score, spf))
            else:
                print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                      (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
        # iou_result[i] = overlap_ratio(gt[i], result_bb[i])[0]

    fps = len(img_list) / spf_total
    # fps = (len(img_list)-1) / spf_total #no first frame
    return iou_result, result_bb, fps, result, object_num_list


def run_mdnet_check(img_list, init_bbox, gt=None, seq='seq_name ex)Basketball', savefig_dir='', display=False,
                    history_step=15, nms_overlap=0.3, imgs=None,

                    object_check_time=10, object_check_threshold=50, object_check_time_num=20,
                    # border_check_time=10, border_check_threshold=20,
                    motion_threshold=5):
    ############################################
    ############################################
    ############################################
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))

    iou_result = np.zeros((len(img_list), 1))
    score_result = np.zeros((len(img_list), 1))

    result[0] = np.copy(target_bbox)
    result_bb[0] = np.copy(target_bbox)

    # execution time array
    exec_time_result = np.zeros((len(img_list), 1))

    # Init model
    model = MDNet(opts['model_path'])
    if opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
    if opts['use_gpu']:
        model = model.cuda()

    # Init image crop model
    img_crop_model = imgCropper(1.)
    if opts['use_gpu']:
        img_crop_model.gpuEnable()

    # Init criterion and optimizer
    criterion = BinaryLoss()

    tic = time.time()
    # Load first image
    if imgs is not None:
        cur_image = imgs[0]
    else:
        cur_image = Image.open(img_list[0]).convert('RGB')
    cur_image = np.asarray(cur_image)

    # init fc and collect traing example
    # model.set_learnable_params(opts['ft_layers'])
    # init_optimizer = set_optimizer(model, opts['lr_init'])
    # pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion, init_optimizer)

    # init backbone
    model.set_all_params_learnable()
    init_backbone_optimizer = set_optimizer(model, 0.001)
    pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion,
                                                           init_backbone_optimizer)

    # _, _, _ = initial_train_with_fam(model, cur_image, target_bbox, img_crop_model, criterion, init_backbone_optimizer)

    model.set_learnable_params(opts['ft_layers'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    spf_total = time.time() - tic
    # spf_total = 0. # no first frame

    # Display
    savefig = savefig_dir != ''
    if display or savefig:

        draw_img = cur_image.copy()
        if gt is not None:
            pt1 = np.around([gt[0, 0], gt[0, 1]]).astype(np.int)
            pt2 = np.around([gt[0, 0] + gt[0, 2], gt[0, 1] + gt[0, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

        pt1 = np.around([result_bb[0, 0], result_bb[0, 1]]).astype(np.int)
        pt2 = np.around([result_bb[0, 0] + result_bb[0, 2], result_bb[0, 1] + result_bb[0, 3]]).astype(np.int)
        cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

        cv2.imshow('show_result', draw_img)
        cv2.waitKey(0)

    # Main loop
    trans_f = opts['trans_f']

    last_target_bbox = target_bbox
    # offset = 0
    motion_model = MotionModeler(histoty_step=history_step)
    graph_match = GraphMatch(nms_overlap=nms_overlap, draw_graph_flag=display)

    begin = None
    motion_cnt = 0

    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        if imgs is not None:
            cur_image = imgs[i]
        else:
            cur_image = Image.open(img_list[i]).convert('RGB')
        cur_image = np.asarray(cur_image)

        # Estimate target bbox
        ishape = cur_image.shape
        # offset = target_bbox - last_target_bbox
        last_target_bbox = target_bbox

        if history_step < 1:
            samples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
                target_bbox, opts['n_samples'])
            samples_weight = None
            motion_predict_rect = None
        else:
            samples, samples_weight, motion_predict_rect = motion_model.generate_samples_fast(cur_image.shape[:-1],
                                                                                              trans_f, 1, True,
                                                                                              target_bbox, 200,
                                                                                              display=display,
                                                                                              img=cur_image.copy(),
                                                                                              ind=i)

        # samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
        #                       target_bbox, opts['n_samples'])

        padded_x1 = (samples[:, 0] - samples[:, 2] * (opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_image.shape[1]:
            padded_scene_box[0] = cur_image.shape[1] - 1
        if padded_scene_box[1] > cur_image.shape[0]:
            padded_scene_box[1] = cur_image.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (padded_scene_box[2:4] * ((opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype(
            'int64')

        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(padded_scene_box, (1, 4)),
                                                                 crop_img_size)
        cropped_image = cropped_image - 128.
        model.eval()
        feat_map = model(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(target_bbox)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, model.receptive_field, (opts['img_size'], opts['img_size']),
                                      target_bbox[2:4], opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = model(sample_feats, in_layer='fc4')
        # sample_scores = torch.nn.functional.sigmoid(sample_scores)
        sample_scores = sample_scores.cpu()
        # if samples_weight is not None:
        #     sample_scores[:, 1] *= torch.from_numpy(samples_weight.astype(np.float32))

        match_rect = graph_match.update(torch.from_numpy(samples), sample_scores[:, 1],
                                        motion_predict_rect if motion_predict_rect is not None else target_bbox,
                                        cur_image.copy() if display else None, i=i)

        if match_rect is None:
            top_scores, top_idx = sample_scores[:, 1].topk(1)
            top_idx = top_idx.data.cpu().numpy()
            target_score = top_scores.data.mean()

            success = target_score > opts['success_thr']
            # if success:
            target_bbox = samples[top_idx].mean(axis=0)
        else:
            target_bbox = match_rect[0]
            target_score = match_rect[1]
            success = target_score > opts['success_thr']

        # # Expand search area at failure
        if success:
            trans_f = opts['trans_f']
        else:
            trans_f = opts['trans_f_expand']

        use_motion = False
        ## Bbox regression
        if success:
            # bbreg_feats = sample_feats[top_idx, :]
            # bbreg_samples = samples[top_idx]
            # bbreg_samples = bbreg.predict(bbreg_feats.data, bbreg_samples)
            # bbreg_bbox = bbreg_samples.mean(axis=0)
            bbreg_bbox = target_bbox  # + offset
            motion_model.update(target_bbox)
            target_score = target_score.sigmoid().detach().cpu().numpy()
            motion_cnt = 0

        else:
            # print('target bbox:', target_bbox)
            # print('motion predict:', motion_predict_rect)

            if motion_predict_rect is not None:
                target_bbox = motion_predict_rect
                use_motion = True
            motion_cnt += 1
            target_score = -1

            bbreg_bbox = target_bbox  # + offset
        # clip bbox
        target_bbox[0] = max(0, target_bbox[0])
        target_bbox[1] = max(0, target_bbox[1])
        target_bbox[0] = min(target_bbox[0], ishape[1] - target_bbox[2])
        target_bbox[1] = min(target_bbox[1], ishape[0] - target_bbox[3])

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox
        iou_result[i] = 1.
        score_result[i] = target_score

        # Data collect
        if success:

            # Draw pos/neg samples
            pos_examples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
                opts['n_pos_update'],
                opts['overlap_pos_update'])
            neg_examples = gen_samples(
                SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
                opts['n_neg_update'],
                opts['overlap_neg_update'])

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
            padded_scene_box = np.reshape(
                np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)), (1, 4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * (
                        (opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype('int64') * jitter_scale[
                                    bidx]
                cropped_image, cur_image_var = img_crop_model.crop_image(cur_image,
                                                                         np.reshape(scene_boxes[bidx], (1, 4)),
                                                                         crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = model(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(target_bbox)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0],
                                                  axis=0)
                scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats  ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > opts['n_pos_update']:
                pos_idx = np.asarray(range(pos_feats.size(0)))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > opts['n_neg_update']:
                neg_idx = np.asarray(range(neg_feats.size(0)))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())

            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)

            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # if i < 200:
        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            # print('## {} ##'.format(i))
            # print(i, success.item())
            draw_img = cur_image.copy()
            all_rect_img = cur_image.copy()
            all_rect_img = cv2.resize(all_rect_img, None, fx=4, fy=4)
            result_img = all_rect_img.copy()
            if gt is not None:
                pt1 = np.around([gt[i, 0], gt[i, 1]]).astype(np.int)
                pt2 = np.around([gt[i, 0] + gt[i, 2], gt[i, 1] + gt[i, 3]]).astype(np.int)
                cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

            pt1 = np.around([result_bb[i, 0], result_bb[i, 1]]).astype(np.int)
            pt2 = np.around([result_bb[i, 0] + result_bb[i, 2], result_bb[i, 1] + result_bb[i, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if use_motion:
                cv2.circle(draw_img, (10, 10), 5, (0, 0, 255), -1)

            # show success potential bbox
            # top_k = 10
            # show_ind = np.argsort(sample_scores[:, 1].detach().numpy())[-top_k:]
            # for ind in show_ind:
            #     bbox = samples[ind]
            #     # print(sample_scores[ind])
            #     pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
            #     pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
            #         np.int)
            #     cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

            for score, bbox in zip(sample_scores[:, 1], samples):
                if score > opts['success_thr']:
                    pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
                    pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
                        np.int)
                    cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

                pt1 = np.around([bbox[0] * 4, bbox[1] * 4]).astype(np.int)
                pt2 = np.around([(bbox[0] + bbox[2]) * 4, (bbox[1] + bbox[3]) * 4]).astype(
                    np.int)
                cv2.rectangle(all_rect_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            pt1 = np.around([result_bb[i, 0] * 4, result_bb[i, 1] * 4]).astype(np.int)
            pt2 = np.around([(result_bb[i, 0] + result_bb[i, 2]) * 4, (result_bb[i, 1] + result_bb[i, 3]) * 4]).astype(
                np.int)
            cv2.rectangle(result_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if motion_predict_rect is not None:
                cx, cy = np.around(motion_predict_rect[:2] + motion_predict_rect[2:] / 2).astype(np.int)
                cv2.circle(draw_img, (cx, cy), 2, (0, 0, 255), -1)

            cv2.imshow('show_result', draw_img)
            # cv2.imwrite(os.path.join('flow_pic/raw', '{:04d}.png'.format(i)), cur_image)
            # cv2.imwrite(os.path.join('flow_pic/candidate', '{:04d}.png'.format(i)), all_rect_img)
            # cv2.imwrite(os.path.join('flow_pic/result', '{:04d}.png'.format(i)), result_img)
            if i < 500:
                cv2.waitKey(1)
            else:
                cv2.waitKey(0)

        if opts['visual_log']:
            if gt is None:
                # print("Frame %d/%d, Score %.3f, Time %.3f" % \
                #       (i, len(img_list), target_score, spf))
                pass
            else:
                # print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                #       (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
                pass
        # iou_result[i] = overlap_ratio(gt[i], result_bb[i])[0]

        if i == object_check_time_num:
            init_pos = [result_bb[0, 0] + result_bb[0, 2] / 2, result_bb[0, 1] + result_bb[0, 3] / 2]
            current_pos = [result_bb[i, 0] + result_bb[i, 2] / 2, result_bb[i, 1] + result_bb[i, 3] / 2]

            displacement = np.sqrt((init_pos[0] - current_pos[0]) ** 2 + (init_pos[1] - current_pos[1]) ** 2)
            print('displacement: ', displacement)
            if displacement < object_check_threshold:
                print('!!!!!!!!!!!!!!stop because of first 10 displacement check!!!!!!!!!!!!!!!!!!!!!!!!!')
                return None, None, None, None, None, i, -1

        if motion_cnt > motion_threshold:
            fps = len(img_list) / spf_total
            print('*****************stop because of first 10 displacement check*')
            return iou_result, result_bb, fps, result, score_result, i - (motion_cnt-1), -2

        # if i > border_check_time:
        #     current_pos = [result_bb[i, 0] + result_bb[i, 2] / 2, result_bb[i, 1] + result_bb[i, 3] / 2]
        #     H, W = ishape[:2]
        #     if current_pos[0] < border_check_threshold or current_pos[0] > (W-border_check_threshold) or current_pos[1] < border_check_threshold or current_pos[1] > (H-border_check_threshold):
        #         fps = len(img_list) / spf_total
        #         # fps = (len(img_list)-1) / spf_total #no first frame
        #         if i > object_check_time_num:
        #             return iou_result, result_bb, fps, result, score_result, i

        if i > 0 and i % object_check_time == 0:
            if i >= object_check_time_num:
                object_pos_info = np.zeros((object_check_time_num, 2))
                for frame_add in range(1, object_check_time_num + 1):
                    object_pos_info[-frame_add, 0] = result_bb[i-frame_add, 0] + result_bb[i-frame_add, 2] / 2
                    object_pos_info[-frame_add, 1] = result_bb[i-frame_add, 1] + result_bb[i-frame_add, 3] / 2
                object_pos_avg = np.mean(object_pos_info, axis=0)

                if begin is None:
                    # begin = [object_tracklet[key][0] + object_tracklet[key][2] / 2,
                    #          object_tracklet[key][1] + object_tracklet[key][3] / 2]
                    begin = object_pos_avg.copy()
                else:
                    # end = [object_tracklet[key][0] + object_tracklet[key][2] / 2,
                    #          object_tracklet[key][1] + object_tracklet[key][3] / 2]
                    end = object_pos_avg.copy()
                    displacement = np.sqrt((begin[0] - end[0]) ** 2 + (begin[1] - end[1]) ** 2)
                    if displacement >= object_check_threshold:
                        begin = object_pos_avg.copy()
                    else:
                        # Todo: this is a logic error, when tracklet is judged as unnormal,
                        # Todo: the previous tracklet should be delete.
                        fps = len(img_list) / spf_total
                        print('!!!!!!!!!!!!!!stop because of displacement check!!!!!!!!!!!!!!!!!!!!!!!!!')
                        return iou_result, result_bb, fps, result, score_result, i, -3

    fps = len(img_list) / spf_total
    # fps = (len(img_list)-1) / spf_total #no first frame
    return iou_result, result_bb, fps, result, score_result, i, 0


def density(traj):
    all_num = 0
    density_num = 0
    for begin in range(0, traj.shape[0] - 1):
        for end in range(1, traj.shape[0]):
            dis_be =  np.sqrt((traj[end, 0] - traj[begin, 0]) ** 2 + (traj[end, 1]-traj[begin, 1])**2)
            for j in range(begin+1, end):
                all_num += 1
                dis_bj = np.sqrt((traj[j, 0] - traj[begin, 0]) ** 2 + (traj[j, 1] - traj[begin, 1]) ** 2)
                if dis_bj > dis_be:
                    density_num += 1
    return density_num / all_num


def run_mdnet_check_density1(img_list, init_bbox, gt=None, seq='seq_name ex)Basketball', savefig_dir='', display=False,
                    history_step=15, nms_overlap=0.3, imgs=None,
                    object_check_time=10, object_check_threshold=50, object_check_time_num=20,
                    # border_check_time=10, border_check_threshold=20,
                    motion_threshold=5):
    ############################################
    ############################################
    ############################################
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))

    iou_result = np.zeros((len(img_list), 1))
    score_result = np.zeros((len(img_list), 1))

    result[0] = np.copy(target_bbox)
    result_bb[0] = np.copy(target_bbox)

    # execution time array
    exec_time_result = np.zeros((len(img_list), 1))

    # Init model
    model = MDNet(opts['model_path'])
    if opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
    if opts['use_gpu']:
        model = model.cuda()

    # Init image crop model
    img_crop_model = imgCropper(1.)
    if opts['use_gpu']:
        img_crop_model.gpuEnable()

    # Init criterion and optimizer
    criterion = BinaryLoss()

    tic = time.time()
    # Load first image
    if imgs is not None:
        cur_image = imgs[0]
    else:
        cur_image = Image.open(img_list[0]).convert('RGB')
    cur_image = np.asarray(cur_image)

    # init fc and collect traing example
    # model.set_learnable_params(opts['ft_layers'])
    # init_optimizer = set_optimizer(model, opts['lr_init'])
    # pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion, init_optimizer)

    # init backbone
    model.set_all_params_learnable()
    init_backbone_optimizer = set_optimizer(model, 0.001)
    pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion,
                                                           init_backbone_optimizer)

    # _, _, _ = initial_train_with_fam(model, cur_image, target_bbox, img_crop_model, criterion, init_backbone_optimizer)

    model.set_learnable_params(opts['ft_layers'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    spf_total = time.time() - tic
    # spf_total = 0. # no first frame

    # Display
    savefig = savefig_dir != ''
    if display or savefig:

        draw_img = cur_image.copy()
        if gt is not None:
            pt1 = np.around([gt[0, 0], gt[0, 1]]).astype(np.int)
            pt2 = np.around([gt[0, 0] + gt[0, 2], gt[0, 1] + gt[0, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

        pt1 = np.around([result_bb[0, 0], result_bb[0, 1]]).astype(np.int)
        pt2 = np.around([result_bb[0, 0] + result_bb[0, 2], result_bb[0, 1] + result_bb[0, 3]]).astype(np.int)
        cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

        cv2.imshow('show_result', draw_img)
        cv2.waitKey(0)

    # Main loop
    trans_f = opts['trans_f']

    last_target_bbox = target_bbox
    # offset = 0
    motion_model = MotionModeler(histoty_step=history_step)
    graph_match = GraphMatch(nms_overlap=nms_overlap, draw_graph_flag=display)

    begin = None
    motion_cnt = 0

    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        if imgs is not None:
            cur_image = imgs[i]
        else:
            cur_image = Image.open(img_list[i]).convert('RGB')
        cur_image = np.asarray(cur_image)

        # Estimate target bbox
        ishape = cur_image.shape
        # offset = target_bbox - last_target_bbox
        last_target_bbox = target_bbox

        if history_step < 1:
            samples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
                target_bbox, opts['n_samples'])
            samples_weight = None
            motion_predict_rect = None
        else:
            samples, samples_weight, motion_predict_rect = motion_model.generate_samples_fast(cur_image.shape[:-1],
                                                                                              trans_f, 1, True,
                                                                                              target_bbox, 200,
                                                                                              display=display,
                                                                                              img=cur_image.copy(),
                                                                                              ind=i)

        # samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
        #                       target_bbox, opts['n_samples'])

        padded_x1 = (samples[:, 0] - samples[:, 2] * (opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_image.shape[1]:
            padded_scene_box[0] = cur_image.shape[1] - 1
        if padded_scene_box[1] > cur_image.shape[0]:
            padded_scene_box[1] = cur_image.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (padded_scene_box[2:4] * ((opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype(
            'int64')

        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(padded_scene_box, (1, 4)),
                                                                 crop_img_size)
        cropped_image = cropped_image - 128.
        model.eval()
        feat_map = model(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(target_bbox)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, model.receptive_field, (opts['img_size'], opts['img_size']),
                                      target_bbox[2:4], opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = model(sample_feats, in_layer='fc4')
        # sample_scores = torch.nn.functional.sigmoid(sample_scores)
        sample_scores = sample_scores.cpu()
        # if samples_weight is not None:
        #     sample_scores[:, 1] *= torch.from_numpy(samples_weight.astype(np.float32))

        match_rect = graph_match.update(torch.from_numpy(samples), sample_scores[:, 1],
                                        motion_predict_rect if motion_predict_rect is not None else target_bbox,
                                        cur_image.copy() if display else None, i=i)

        if match_rect is None:
            top_scores, top_idx = sample_scores[:, 1].topk(1)
            top_idx = top_idx.data.cpu().numpy()
            target_score = top_scores.data.mean()

            success = target_score > opts['success_thr']
            # if success:
            target_bbox = samples[top_idx].mean(axis=0)
        else:
            target_bbox = match_rect[0]
            target_score = match_rect[1]
            success = target_score > opts['success_thr']

        # # Expand search area at failure
        if success:
            trans_f = opts['trans_f']
        else:
            trans_f = opts['trans_f_expand']

        use_motion = False
        ## Bbox regression
        if success:
            # bbreg_feats = sample_feats[top_idx, :]
            # bbreg_samples = samples[top_idx]
            # bbreg_samples = bbreg.predict(bbreg_feats.data, bbreg_samples)
            # bbreg_bbox = bbreg_samples.mean(axis=0)
            bbreg_bbox = target_bbox  # + offset
            motion_model.update(target_bbox)
            target_score = target_score.sigmoid().detach().cpu().numpy()
            motion_cnt = 0

        else:
            # print('target bbox:', target_bbox)
            # print('motion predict:', motion_predict_rect)

            if motion_predict_rect is not None:
                target_bbox = motion_predict_rect
                use_motion = True
            motion_cnt += 1
            target_score = -1

            bbreg_bbox = target_bbox  # + offset
        # clip bbox
        target_bbox[0] = max(0, target_bbox[0])
        target_bbox[1] = max(0, target_bbox[1])
        target_bbox[0] = min(target_bbox[0], ishape[1] - target_bbox[2])
        target_bbox[1] = min(target_bbox[1], ishape[0] - target_bbox[3])

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox
        iou_result[i] = 1.
        score_result[i] = target_score

        # Data collect
        if success:

            # Draw pos/neg samples
            pos_examples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
                opts['n_pos_update'],
                opts['overlap_pos_update'])
            neg_examples = gen_samples(
                SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
                opts['n_neg_update'],
                opts['overlap_neg_update'])

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
            padded_scene_box = np.reshape(
                np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)), (1, 4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * (
                        (opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype('int64') * jitter_scale[
                                    bidx]
                cropped_image, cur_image_var = img_crop_model.crop_image(cur_image,
                                                                         np.reshape(scene_boxes[bidx], (1, 4)),
                                                                         crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = model(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(target_bbox)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0],
                                                  axis=0)
                scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats  ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > opts['n_pos_update']:
                pos_idx = np.asarray(range(pos_feats.size(0)))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > opts['n_neg_update']:
                neg_idx = np.asarray(range(neg_feats.size(0)))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())

            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)

            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # if i < 200:
        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            # print('## {} ##'.format(i))
            # print(i, success.item())
            draw_img = cur_image.copy()
            all_rect_img = cur_image.copy()
            all_rect_img = cv2.resize(all_rect_img, None, fx=4, fy=4)
            result_img = all_rect_img.copy()
            if gt is not None:
                pt1 = np.around([gt[i, 0], gt[i, 1]]).astype(np.int)
                pt2 = np.around([gt[i, 0] + gt[i, 2], gt[i, 1] + gt[i, 3]]).astype(np.int)
                cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

            pt1 = np.around([result_bb[i, 0], result_bb[i, 1]]).astype(np.int)
            pt2 = np.around([result_bb[i, 0] + result_bb[i, 2], result_bb[i, 1] + result_bb[i, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if use_motion:
                cv2.circle(draw_img, (10, 10), 5, (0, 0, 255), -1)

            # show success potential bbox
            # top_k = 10
            # show_ind = np.argsort(sample_scores[:, 1].detach().numpy())[-top_k:]
            # for ind in show_ind:
            #     bbox = samples[ind]
            #     # print(sample_scores[ind])
            #     pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
            #     pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
            #         np.int)
            #     cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

            for score, bbox in zip(sample_scores[:, 1], samples):
                if score > opts['success_thr']:
                    pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
                    pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
                        np.int)
                    cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

                pt1 = np.around([bbox[0] * 4, bbox[1] * 4]).astype(np.int)
                pt2 = np.around([(bbox[0] + bbox[2]) * 4, (bbox[1] + bbox[3]) * 4]).astype(
                    np.int)
                cv2.rectangle(all_rect_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            pt1 = np.around([result_bb[i, 0] * 4, result_bb[i, 1] * 4]).astype(np.int)
            pt2 = np.around([(result_bb[i, 0] + result_bb[i, 2]) * 4, (result_bb[i, 1] + result_bb[i, 3]) * 4]).astype(
                np.int)
            cv2.rectangle(result_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if motion_predict_rect is not None:
                cx, cy = np.around(motion_predict_rect[:2] + motion_predict_rect[2:] / 2).astype(np.int)
                cv2.circle(draw_img, (cx, cy), 2, (0, 0, 255), -1)

            cv2.imshow('show_result', draw_img)
            # cv2.imwrite(os.path.join('flow_pic/raw', '{:04d}.png'.format(i)), cur_image)
            # cv2.imwrite(os.path.join('flow_pic/candidate', '{:04d}.png'.format(i)), all_rect_img)
            # cv2.imwrite(os.path.join('flow_pic/result', '{:04d}.png'.format(i)), result_img)
            if i < 500:
                cv2.waitKey(1)
            else:
                cv2.waitKey(0)

        if opts['visual_log']:
            if gt is None:
                # print("Frame %d/%d, Score %.3f, Time %.3f" % \
                #       (i, len(img_list), target_score, spf))
                pass
            else:
                # print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                #       (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
                pass
        # iou_result[i] = overlap_ratio(gt[i], result_bb[i])[0]

        if i == object_check_time_num:
            init_pos = [result_bb[0, 0] + result_bb[0, 2] / 2, result_bb[0, 1] + result_bb[0, 3] / 2]
            current_pos = [result_bb[i, 0] + result_bb[i, 2] / 2, result_bb[i, 1] + result_bb[i, 3] / 2]

            displacement = np.sqrt((init_pos[0] - current_pos[0]) ** 2 + (init_pos[1] - current_pos[1]) ** 2)
            print('displacement: ', displacement)
            if displacement < object_check_threshold:
                print('!!!!!!!!!!!!!!stop because of first 10 displacement check!!!!!!!!!!!!!!!!!!!!!!!!!')
                return None, None, None, None, None, i, -1

        if motion_cnt > motion_threshold:
            fps = len(img_list) / spf_total
            print('*****************stop because of first 10 displacement check*')
            return iou_result, result_bb, fps, result, score_result, i - (motion_cnt-1), -2

        # if i > border_check_time:
        #     current_pos = [result_bb[i, 0] + result_bb[i, 2] / 2, result_bb[i, 1] + result_bb[i, 3] / 2]
        #     H, W = ishape[:2]
        #     if current_pos[0] < border_check_threshold or current_pos[0] > (W-border_check_threshold) or current_pos[1] < border_check_threshold or current_pos[1] > (H-border_check_threshold):
        #         fps = len(img_list) / spf_total
        #         # fps = (len(img_list)-1) / spf_total #no first frame
        #         if i > object_check_time_num:
        #             return iou_result, result_bb, fps, result, score_result, i

        if i > 0 and i % object_check_time == 0:
            if i >= object_check_time_num:
                object_pos_info = np.zeros((object_check_time_num, 2))
                for frame_add in range(1, object_check_time_num + 1):
                    object_pos_info[-frame_add, 0] = result_bb[i-frame_add, 0] + result_bb[i-frame_add, 2] / 2
                    object_pos_info[-frame_add, 1] = result_bb[i-frame_add, 1] + result_bb[i-frame_add, 3] / 2
                object_pos_avg = np.mean(object_pos_info, axis=0)

                if begin is None:
                    # begin = [object_tracklet[key][0] + object_tracklet[key][2] / 2,
                    #          object_tracklet[key][1] + object_tracklet[key][3] / 2]
                    begin = object_pos_avg.copy()
                else:
                    # end = [object_tracklet[key][0] + object_tracklet[key][2] / 2,
                    #          object_tracklet[key][1] + object_tracklet[key][3] / 2]
                    end = object_pos_avg.copy()
                    displacement = np.sqrt((begin[0] - end[0]) ** 2 + (begin[1] - end[1]) ** 2)
                    if displacement >= object_check_threshold:
                        begin = object_pos_avg.copy()
                    else:
                        # Todo: this is a logic error, when tracklet is judged as unnormal,
                        # Todo: the previous tracklet should be delete.
                        fps = len(img_list) / spf_total
                        print('!!!!!!!!!!!!!!stop because of displacement check!!!!!!!!!!!!!!!!!!!!!!!!!')
                        return iou_result, result_bb, fps, result, score_result, i, -3

    fps = len(img_list) / spf_total
    # fps = (len(img_list)-1) / spf_total #no first frame
    return iou_result, result_bb, fps, result, score_result, i, 0

def run_mdnet_check_density(img_list, init_bbox, gt=None, seq='seq_name ex)Basketball', savefig_dir='', display=False,
                    history_step=15, nms_overlap=0.3, imgs=None,
                    object_check_time=10, object_check_threshold=50, object_check_time_num=20,
                    border_check_time=10, border_check_threshold=20,
                    motion_threshold=5):
    ############################################
    ############################################
    ############################################
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))

    iou_result = np.zeros((len(img_list), 1))
    score_result = np.zeros((len(img_list), 1))

    result[0] = np.copy(target_bbox)
    result_bb[0] = np.copy(target_bbox)

    # execution time array
    exec_time_result = np.zeros((len(img_list), 1))

    # Init model
    model = MDNet(opts['model_path'])
    if opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
    if opts['use_gpu']:
        model = model.cuda()

    # Init image crop model
    img_crop_model = imgCropper(1.)
    if opts['use_gpu']:
        img_crop_model.gpuEnable()

    # Init criterion and optimizer
    criterion = BinaryLoss()

    tic = time.time()
    # Load first image
    if imgs is not None:
        cur_image = imgs[0]
    else:
        cur_image = Image.open(img_list[0]).convert('RGB')
    cur_image = np.asarray(cur_image)

    # init fc and collect traing example
    # model.set_learnable_params(opts['ft_layers'])
    # init_optimizer = set_optimizer(model, opts['lr_init'])
    # pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion, init_optimizer)

    # init backbone
    model.set_all_params_learnable()
    init_backbone_optimizer = set_optimizer(model, 0.001)
    pos_feats_all, neg_feats_all, feat_dim = initial_train(model, cur_image, target_bbox, img_crop_model, criterion,
                                                           init_backbone_optimizer)

    # _, _, _ = initial_train_with_fam(model, cur_image, target_bbox, img_crop_model, criterion, init_backbone_optimizer)

    model.set_learnable_params(opts['ft_layers'])
    update_optimizer = set_optimizer(model, opts['lr_update'])

    spf_total = time.time() - tic
    # spf_total = 0. # no first frame

    # Display
    savefig = savefig_dir != ''
    if display or savefig:

        draw_img = cur_image.copy()
        if gt is not None:
            pt1 = np.around([gt[0, 0], gt[0, 1]]).astype(np.int)
            pt2 = np.around([gt[0, 0] + gt[0, 2], gt[0, 1] + gt[0, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

        pt1 = np.around([result_bb[0, 0], result_bb[0, 1]]).astype(np.int)
        pt2 = np.around([result_bb[0, 0] + result_bb[0, 2], result_bb[0, 1] + result_bb[0, 3]]).astype(np.int)
        cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

        # cv2.imshow('show_result', draw_img)
        # cv2.waitKey(0)

    # Main loop
    trans_f = opts['trans_f']

    last_target_bbox = target_bbox
    # offset = 0
    motion_model = MotionModeler(histoty_step=history_step)
    graph_match = GraphMatch(nms_overlap=nms_overlap, draw_graph_flag=display)

    begin = None
    motion_cnt = 0

    for i in range(1, len(img_list)):

        tic = time.time()
        # Load image
        if imgs is not None:
            cur_image = imgs[i]
        else:
            cur_image = Image.open(img_list[i]).convert('RGB')
        cur_image = np.asarray(cur_image)

        # Estimate target bbox
        ishape = cur_image.shape
        # offset = target_bbox - last_target_bbox
        last_target_bbox = target_bbox

        if history_step < 1:
            samples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
                target_bbox, opts['n_samples'])
            samples_weight = None
            motion_predict_rect = None
        else:
            samples, samples_weight, motion_predict_rect = motion_model.generate_samples_fast(cur_image.shape[:-1],
                                                                                              trans_f, 1, True,
                                                                                              target_bbox, 200,
                                                                                              display=display,
                                                                                              img=cur_image.copy(),
                                                                                              ind=i)

        # samples = gen_samples(SampleGenerator('gaussian', (ishape[1], ishape[0]), trans_f, opts['scale_f'], valid=True),
        #                       target_bbox, opts['n_samples'])

        padded_x1 = (samples[:, 0] - samples[:, 2] * (opts['padding'] - 1.) / 2.).min()
        padded_y1 = (samples[:, 1] - samples[:, 3] * (opts['padding'] - 1.) / 2.).min()
        padded_x2 = (samples[:, 0] + samples[:, 2] * (opts['padding'] + 1.) / 2.).max()
        padded_y2 = (samples[:, 1] + samples[:, 3] * (opts['padding'] + 1.) / 2.).max()
        padded_scene_box = np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1))

        if padded_scene_box[0] > cur_image.shape[1]:
            padded_scene_box[0] = cur_image.shape[1] - 1
        if padded_scene_box[1] > cur_image.shape[0]:
            padded_scene_box[1] = cur_image.shape[0] - 1
        if padded_scene_box[0] + padded_scene_box[2] < 0:
            padded_scene_box[2] = -padded_scene_box[0] + 1
        if padded_scene_box[1] + padded_scene_box[3] < 0:
            padded_scene_box[3] = -padded_scene_box[1] + 1

        crop_img_size = (padded_scene_box[2:4] * ((opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype(
            'int64')

        cropped_image, cur_image_var = img_crop_model.crop_image(cur_image, np.reshape(padded_scene_box, (1, 4)),
                                                                 crop_img_size)
        cropped_image = cropped_image - 128.
        model.eval()
        feat_map = model(cropped_image, out_layer='conv3')

        # relative target bbox with padded_scene_box
        rel_target_bbox = np.copy(target_bbox)
        rel_target_bbox[0:2] -= padded_scene_box[0:2]

        # Extract sample features and get target location
        batch_num = np.zeros((samples.shape[0], 1))
        sample_rois = np.copy(samples)
        sample_rois[:, 0:2] -= np.repeat(np.reshape(padded_scene_box[0:2], (1, 2)), sample_rois.shape[0], axis=0)
        sample_rois = samples2maskroi(sample_rois, model.receptive_field, (opts['img_size'], opts['img_size']),
                                      target_bbox[2:4], opts['padding'])
        sample_rois = np.concatenate((batch_num, sample_rois), axis=1)
        sample_rois = Variable(torch.from_numpy(sample_rois.astype('float32'))).cuda()
        sample_feats = model.roi_align_model(feat_map, sample_rois)
        sample_feats = sample_feats.view(sample_feats.size(0), -1).clone()
        sample_scores = model(sample_feats, in_layer='fc4')
        # sample_scores = torch.nn.functional.sigmoid(sample_scores)
        sample_scores = sample_scores.cpu()
        # if samples_weight is not None:
        #     sample_scores[:, 1] *= torch.from_numpy(samples_weight.astype(np.float32))

        match_rect = graph_match.update(torch.from_numpy(samples), sample_scores[:, 1],
                                        motion_predict_rect if motion_predict_rect is not None else target_bbox,
                                        cur_image.copy() if display else None, i=i)

        if match_rect is None:
            top_scores, top_idx = sample_scores[:, 1].topk(1)
            top_idx = top_idx.data.cpu().numpy()
            target_score = top_scores.data.mean()

            success = target_score > opts['success_thr']
            # if success:
            target_bbox = samples[top_idx].mean(axis=0)
        else:
            target_bbox = match_rect[0]
            target_score = match_rect[1]
            success = target_score > opts['success_thr']

        # # Expand search area at failure
        if success:
            trans_f = opts['trans_f']
        else:
            trans_f = opts['trans_f_expand']

        use_motion = False
        ## Bbox regression
        if success:
            # bbreg_feats = sample_feats[top_idx, :]
            # bbreg_samples = samples[top_idx]
            # bbreg_samples = bbreg.predict(bbreg_feats.data, bbreg_samples)
            # bbreg_bbox = bbreg_samples.mean(axis=0)
            bbreg_bbox = target_bbox  # + offset
            motion_model.update(target_bbox)
            target_score = target_score.sigmoid().detach().cpu().numpy()
            motion_cnt = 0

        else:
            # print('target bbox:', target_bbox)
            # print('motion predict:', motion_predict_rect)

            if motion_predict_rect is not None:
                target_bbox = motion_predict_rect
                use_motion = True
            motion_cnt += 1
            target_score = -1

            bbreg_bbox = target_bbox  # + offset
        # clip bbox
        target_bbox[0] = max(0, target_bbox[0])
        target_bbox[1] = max(0, target_bbox[1])
        target_bbox[0] = min(target_bbox[0], ishape[1] - target_bbox[2])
        target_bbox[1] = min(target_bbox[1], ishape[0] - target_bbox[3])

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox
        iou_result[i] = 1.
        score_result[i] = target_score

        # Data collect
        if success:

            # Draw pos/neg samples
            pos_examples = gen_samples(
                SampleGenerator('gaussian', (ishape[1], ishape[0]), 0.1, 1.2), target_bbox,
                opts['n_pos_update'],
                opts['overlap_pos_update'])
            neg_examples = gen_samples(
                SampleGenerator('uniform', (ishape[1], ishape[0]), 1.5, 1.2), target_bbox,
                opts['n_neg_update'],
                opts['overlap_neg_update'])

            padded_x1 = (neg_examples[:, 0] - neg_examples[:, 2] * (opts['padding'] - 1.) / 2.).min()
            padded_y1 = (neg_examples[:, 1] - neg_examples[:, 3] * (opts['padding'] - 1.) / 2.).min()
            padded_x2 = (neg_examples[:, 0] + neg_examples[:, 2] * (opts['padding'] + 1.) / 2.).max()
            padded_y2 = (neg_examples[:, 1] + neg_examples[:, 3] * (opts['padding'] + 1.) / 2.).max()
            padded_scene_box = np.reshape(
                np.asarray((padded_x1, padded_y1, padded_x2 - padded_x1, padded_y2 - padded_y1)), (1, 4))

            scene_boxes = np.reshape(np.copy(padded_scene_box), (1, 4))
            jitter_scale = [1.]

            for bidx in range(0, scene_boxes.shape[0]):
                crop_img_size = (scene_boxes[bidx, 2:4] * (
                        (opts['img_size'], opts['img_size']) / target_bbox[2:4])).astype('int64') * jitter_scale[
                                    bidx]
                cropped_image, cur_image_var = img_crop_model.crop_image(cur_image,
                                                                         np.reshape(scene_boxes[bidx], (1, 4)),
                                                                         crop_img_size)
                cropped_image = cropped_image - 128.

                feat_map = model(cropped_image, out_layer='conv3')

                rel_target_bbox = np.copy(target_bbox)
                rel_target_bbox[0:2] -= scene_boxes[bidx, 0:2]

                batch_num = np.zeros((pos_examples.shape[0], 1))
                cur_pos_rois = np.copy(pos_examples)
                cur_pos_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_pos_rois.shape[0],
                                                  axis=0)
                scaled_obj_size = float(opts['img_size']) * jitter_scale[bidx]
                cur_pos_rois = samples2maskroi(cur_pos_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_pos_rois = np.concatenate((batch_num, cur_pos_rois), axis=1)
                cur_pos_rois = Variable(torch.from_numpy(cur_pos_rois.astype('float32'))).cuda()
                cur_pos_feats = model.roi_align_model(feat_map, cur_pos_rois)
                cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1).data.clone()

                batch_num = np.zeros((neg_examples.shape[0], 1))
                cur_neg_rois = np.copy(neg_examples)
                cur_neg_rois[:, 0:2] -= np.repeat(np.reshape(scene_boxes[bidx, 0:2], (1, 2)), cur_neg_rois.shape[0],
                                                  axis=0)
                cur_neg_rois = samples2maskroi(cur_neg_rois, model.receptive_field, (scaled_obj_size, scaled_obj_size),
                                               target_bbox[2:4], opts['padding'])
                cur_neg_rois = np.concatenate((batch_num, cur_neg_rois), axis=1)
                cur_neg_rois = Variable(torch.from_numpy(cur_neg_rois.astype('float32'))).cuda()
                cur_neg_feats = model.roi_align_model(feat_map, cur_neg_rois)
                cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1).data.clone()

                feat_dim = cur_pos_feats.size(-1)

                if bidx == 0:
                    pos_feats = cur_pos_feats  ##index select
                    neg_feats = cur_neg_feats
                else:
                    pos_feats = torch.cat((pos_feats, cur_pos_feats), dim=0)
                    neg_feats = torch.cat((neg_feats, cur_neg_feats), dim=0)

            if pos_feats.size(0) > opts['n_pos_update']:
                pos_idx = np.asarray(range(pos_feats.size(0)))
                np.random.shuffle(pos_idx)
                pos_feats = pos_feats.index_select(0, torch.from_numpy(pos_idx[0:opts['n_pos_update']]).cuda())
            if neg_feats.size(0) > opts['n_neg_update']:
                neg_idx = np.asarray(range(neg_feats.size(0)))
                np.random.shuffle(neg_idx)
                neg_feats = neg_feats.index_select(0, torch.from_numpy(neg_idx[0:opts['n_neg_update']]).cuda())

            pos_feats_all.append(pos_feats)
            neg_feats_all.append(neg_feats)

            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # if i < 200:
        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            # print('## {} ##'.format(i))
            # print(i, success.item())
            draw_img = cur_image.copy()
            all_rect_img = cur_image.copy()
            all_rect_img = cv2.resize(all_rect_img, None, fx=4, fy=4)
            result_img = all_rect_img.copy()
            if gt is not None:
                pt1 = np.around([gt[i, 0], gt[i, 1]]).astype(np.int)
                pt2 = np.around([gt[i, 0] + gt[i, 2], gt[i, 1] + gt[i, 3]]).astype(np.int)
                cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 255, 0), 1)

            pt1 = np.around([result_bb[i, 0], result_bb[i, 1]]).astype(np.int)
            pt2 = np.around([result_bb[i, 0] + result_bb[i, 2], result_bb[i, 1] + result_bb[i, 3]]).astype(np.int)
            cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if use_motion:
                cv2.circle(draw_img, (10, 10), 5, (0, 0, 255), -1)

            # show success potential bbox
            # top_k = 10
            # show_ind = np.argsort(sample_scores[:, 1].detach().numpy())[-top_k:]
            # for ind in show_ind:
            #     bbox = samples[ind]
            #     # print(sample_scores[ind])
            #     pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
            #     pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
            #         np.int)
            #     cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

            for score, bbox in zip(sample_scores[:, 1], samples):
                if score > opts['success_thr']:
                    pt1 = np.around([bbox[0], bbox[1]]).astype(np.int)
                    pt2 = np.around([bbox[0] + bbox[2], bbox[1] + bbox[3]]).astype(
                        np.int)
                    cv2.rectangle(draw_img, tuple(pt1), tuple(pt2), (255, 0, 0), 1)

                pt1 = np.around([bbox[0] * 4, bbox[1] * 4]).astype(np.int)
                pt2 = np.around([(bbox[0] + bbox[2]) * 4, (bbox[1] + bbox[3]) * 4]).astype(
                    np.int)
                cv2.rectangle(all_rect_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            pt1 = np.around([result_bb[i, 0] * 4, result_bb[i, 1] * 4]).astype(np.int)
            pt2 = np.around([(result_bb[i, 0] + result_bb[i, 2]) * 4, (result_bb[i, 1] + result_bb[i, 3]) * 4]).astype(
                np.int)
            cv2.rectangle(result_img, tuple(pt1), tuple(pt2), (0, 0, 255), 1)

            if motion_predict_rect is not None:
                cx, cy = np.around(motion_predict_rect[:2] + motion_predict_rect[2:] / 2).astype(np.int)
                cv2.circle(draw_img, (cx, cy), 2, (0, 0, 255), -1)

            # cv2.imshow('show_result', draw_img)
            # cv2.imwrite(os.path.join('flow_pic/raw', '{:04d}.png'.format(i)), cur_image)
            # cv2.imwrite(os.path.join('flow_pic/candidate', '{:04d}.png'.format(i)), all_rect_img)
            # cv2.imwrite(os.path.join('flow_pic/result', '{:04d}.png'.format(i)), result_img)
            # if i < 500:
            #     cv2.waitKey(1)
            # else:
            #     cv2.waitKey(0)

        if opts['visual_log']:
            if gt is None:
                # print("Frame %d/%d, Score %.3f, Time %.3f" % \
                #       (i, len(img_list), target_score, spf))
                pass
            else:
                # print("Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                #       (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf))
                pass
        # iou_result[i] = overlap_ratio(gt[i], result_bb[i])[0]

        if i == object_check_time_num:
            init_pos = [result_bb[0, 0] + result_bb[0, 2] / 2, result_bb[0, 1] + result_bb[0, 3] / 2]
            current_pos = [result_bb[i, 0] + result_bb[i, 2] / 2, result_bb[i, 1] + result_bb[i, 3] / 2]

            displacement = np.sqrt((init_pos[0] - current_pos[0]) ** 2 + (init_pos[1] - current_pos[1]) ** 2)
            print('displacement: ', displacement)
            if displacement < object_check_threshold:
                print('11111111111111111111111111stop 10 first displacement check111111111111111111111111111111')
                return None, None, None, None, None, i, -1

        if motion_cnt > motion_threshold:
            fps = len(img_list) / spf_total
            print('*******************stop motion check*********************')
            return iou_result, result_bb, fps, result, score_result, i - motion_cnt, -2

        # if i > border_check_time:
        #     current_pos = [result_bb[i, 0] + result_bb[i, 2] / 2, result_bb[i, 1] + result_bb[i, 3] / 2]
        #     H, W = ishape[:2]
        #     if current_pos[0] < border_check_threshold or current_pos[0] > (W-border_check_threshold) or current_pos[1] < border_check_threshold or current_pos[1] > (H-border_check_threshold):
        #         fps = len(img_list) / spf_total
        #         # fps = (len(img_list)-1) / spf_total #no first frame
        #         if i > object_check_time_num:
        #             return iou_result, result_bb, fps, result, score_result, i

        if i > 0 and i % object_check_time == 0:
            if i >= object_check_time_num:
                object_pos_info = np.zeros((object_check_time_num, 2))
                for frame_add in range(1, object_check_time_num + 1):
                    object_pos_info[-frame_add, 0] = result_bb[i-frame_add, 0] + result_bb[i-frame_add, 2] / 2
                    object_pos_info[-frame_add, 1] = result_bb[i-frame_add, 1] + result_bb[i-frame_add, 3] / 2

                object_pos_avg = np.mean(object_pos_info, axis=0)

                traj_den = density(object_pos_info)
                if traj_den > 0.5:
                    fps = len(img_list) / spf_total
                    print('!!!!!!!!!!!!!!stop density check!!!!!!!!!!!!!!!!!!!!!!!!!')
                    return iou_result, result_bb, fps, result, score_result, i, -3

                if begin is None:
                    # begin = [object_tracklet[key][0] + object_tracklet[key][2] / 2,
                    #          object_tracklet[key][1] + object_tracklet[key][3] / 2]
                    begin = object_pos_avg.copy()
                else:
                    # end = [object_tracklet[key][0] + object_tracklet[key][2] / 2,
                    #          object_tracklet[key][1] + object_tracklet[key][3] / 2]
                    end = object_pos_avg.copy()
                    displacement = np.sqrt((begin[0] - end[0]) ** 2 + (begin[1] - end[1]) ** 2)
                    if displacement >= object_check_threshold:
                        begin = object_pos_avg.copy()
                    else:
                        # Todo: this is a logic error, when tracklet is judged as unnormal,
                        # Todo: the previous tracklet should be delete.
                        fps = len(img_list) / spf_total
                        print('+++++++++++++++++stop displacement check+++++++++++++++++++++')
                        return iou_result, result_bb, fps, result, score_result, i, -1

    fps = len(img_list) / spf_total
    # fps = (len(img_list)-1) / spf_total #no first frame
    return iou_result, result_bb, fps, result, score_result, i, 0




