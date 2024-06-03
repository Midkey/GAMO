
from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np
import time
from tracker import *


best = None 
if best is not None:
    with open(best, 'r') as f:
        best_str = f.readlines()
else:
    best_str = None


from util import backup_file


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt in zip(out_res, label_res):
        measure_per_frame.append(iou(_pred, _gt))
    return np.mean(measure_per_frame)


def main(video_root, save_note, tracker_name, dataset_name, visulization=False, set_video_name=''):
    # setup tracker
    opts['model_path'] = './models/rt_mdnet.pth'
    opts['visual_log'] = False


    # setup experiments
    with open(os.path.join(video_root, 'list.txt'), 'r') as f:
        video_paths = f.readlines()
    video_paths = [v.strip() for v in video_paths]#[1:]
    video_num = len(video_paths)

    history_search = [10, ]
    nms_search = [0.2, ]

    iter_time = 1

    for i in range(iter_time):
        for history in history_search:
            for nms in nms_search:
                s = save_note.format(i, history, nms)
                output_dir = os.path.join('result', dataset_name, tracker_name, s)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                backup_file([__file__, 'tracker.py', 'modules/graph_match.py', 'motion_model.py'], output_dir)

    overall_performance = {}

    if use_time:
        pass
    else:
        set_video_name = ''

    # run tracking experiments and report performance
    for video_id, video_name in enumerate(video_paths, start=1):
        print(video_id, video_name)

        if len(set_video_name) > 0 and set_video_name not in video_name:
            continue

        video_path = os.path.join(video_root, video_name, 'img')
        frame_list = os.listdir(video_path)
        frame_list.sort()
        try:
            res_file = os.path.join(video_root, video_name, 'groundtruth_rect.txt')

            label_res = np.loadtxt(res_file, delimiter=',')[0]
            init_rect = label_res
        except:
            res_file = os.path.join(video_root, video_name, 'groundtruth.txt')
            with open(res_file, 'r') as f:
                info = f.readline()
                init_rect = list(map(float, info.strip().split(',')))

        total_time = 0
        image_list = []

        for frame_id, frame_name in enumerate(frame_list):
            image_list.append(os.path.join(video_path, frame_name))

        imgs = load_img(image_list)

        for i in range(iter_time):
            for history in history_search:
                for nms in nms_search:
                    s = save_note.format(i, history, nms)
                    output_dir = os.path.join('result', dataset_name, tracker_name, s)
                    # print('[{}]'.format(i) + s)

                    start_time = time.time()
                    # try:
                    iou_result, result_bb, fps, result_nobb, object_num_list = run_mdnet(image_list, init_rect, None, seq =video_name, display=visulization,
                                                                        history_step=history, nms_overlap=nms, imgs=imgs)
                    # except BaseException as e:
                    #     print(s, e)
                    #     continue
                    # save result
                    spend_time = time.time() - start_time
                    output_file = os.path.join(output_dir, '%s_time.txt' % (video_name))
                    with open(output_file, 'w') as f:
                        f.write('{}\n{}\n{}\n'.format(spend_time, len(result_bb), len(result_bb) / spend_time))

                    output_file = os.path.join(output_dir, '%s.txt' % (video_name))
                    np.savetxt(output_file, np.array(result_nobb), delimiter=',')

                    # mixed_measure = eval(result_nobb, label_res)
                    # if s in overall_performance:
                    #     overall_performance[s].append(mixed_measure)
                    # else:
                    #     overall_performance[s] = [mixed_measure]
                    # print_str = '[%03d/%03d] %20s  Fixed Measure: %.03f. FPS: %.04f' % (video_id, video_num, video_name, mixed_measure, fps)
                    # print(print_str)
                    # with open(os.path.join(output_dir, 'log.txt'), 'a') as f:
                    #     f.write(print_str + '\n')
    for i in range(iter_time):
        for history in history_search:
            for nms in nms_search:
                s = save_note.format(i, history, nms)
                if s not in overall_performance:
                    continue
                print_str = '[Overall] Mixed Measure: %.03f\n  ' % (np.mean(overall_performance[s]))
                with open(os.path.join('result_vatsot', dataset_name, tracker_name, s, 'log.txt'), 'a') as f:
                    f.write(print_str + '\n')

def load_img(img_list):
    imgs = []
    import tqdm
    for im_name in tqdm.tqdm(img_list):
        im = Image.open(im_name).convert('RGB')
        imgs.append(im)
    return imgs


if __name__ == '__main__':
    root = '' # dataset root
    tracker_name = 'GAMO'
    use_time = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    dataset_list = ['SV248S']
    set_name = ''

    for dataset_name in dataset_list:
        print(dataset_name)
        if use_time:
            save_note += time.strftime(fmt, time.localtime(time.time()))
        else:
            save_note += '-debug'

        video_root = os.path.join(root, dataset_name)

        save_note = tracker_name + '{}-_RT-motion_enlarge-{:02d}-graph_match-{}'
        fmt = ' --- %Y-%m-%d %a %H:%M'  # 格式化时间
        save_note += time.strftime(fmt, time.localtime(time.time()))
        main(video_root, save_note, tracker_name, dataset_name, visulization=not use_time, set_video_name=set_name)

