#from python.keypoint_preprocess import expand_crop
#from python.visualize import visualize_pose


import os
import cv2
import numpy as np
import math

def expand_crop(images, rect, expand_ratio=0.3):
    imgh, imgw, c = images.shape
    label, conf, xmin, ymin, xmax, ymax = [int(x) for x in rect.tolist()]
    if label != 0:
        return None, None, None
    org_rect = [xmin, ymin, xmax, ymax]
    h_half = (ymax - ymin) * (1 + expand_ratio) / 2.
    w_half = (xmax - xmin) * (1 + expand_ratio) / 2.
    if h_half > w_half * 4 / 3:
        w_half = h_half * 0.75
    center = [(ymin + ymax) / 2., (xmin + xmax) / 2.]
    ymin = max(0, int(center[0] - h_half))
    ymax = min(imgh - 1, int(center[0] + h_half))
    xmin = max(0, int(center[1] - w_half))
    xmax = min(imgw - 1, int(center[1] + w_half))
    return images[ymin:ymax, xmin:xmax, :], [xmin, ymin, xmax, ymax], org_rect


def visualize_pose(imgfile,
                   results,
                   visual_thresh=0.6,
                   save_name='pose.jpg',
                   save_dir='output',
                   returnimg=False,
                   ids=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        plt.switch_backend('agg')
    except Exception as e:
        print('Matplotlib not found, please install matplotlib.'
              'for example: `pip install matplotlib`.')
        raise e
    skeletons, scores = results['keypoint']
    skeletons = np.array(skeletons)
    kpt_nums = 17
    if len(skeletons) > 0:
        kpt_nums = skeletons.shape[1]
    if kpt_nums == 17:  #plot coco keypoint
        EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
                 (7, 9), (8, 10), (5, 11), (6, 12), (11, 13), (12, 14),
                 (13, 15), (14, 16), (11, 12)]
    else:  #plot mpii keypoint
        EDGES = [(0, 1), (1, 2), (3, 4), (4, 5), (2, 6), (3, 6), (6, 7), (7, 8),
                 (8, 9), (10, 11), (11, 12), (13, 14), (14, 15), (8, 12),
                 (8, 13)]
    NUM_EDGES = len(EDGES)

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    cmap = matplotlib.cm.get_cmap('hsv')
    plt.figure()

    img = cv2.imread(imgfile) if type(imgfile) == str else imgfile

    color_set = results['colors'] if 'colors' in results else None

    if 'bbox' in results and ids is None:
        bboxs = results['bbox']
        for j, rect in enumerate(bboxs):
            xmin, ymin, xmax, ymax = rect
            color = colors[0] if color_set is None else colors[color_set[j] %
                                                               len(colors)]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)

    canvas = img.copy()
    for i in range(kpt_nums):
        for j in range(len(skeletons)):
            if skeletons[j][i, 2] < visual_thresh:
                continue
            if ids is None:
                color = colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])

            cv2.circle(
                canvas,
                tuple(skeletons[j][i, 0:2].astype('int32')),
                2,
                color,
                thickness=-1)

    to_plot = cv2.addWeighted(img, 0.3, canvas, 0.7, 0)
    fig = matplotlib.pyplot.gcf()

    stickwidth = 2

    for i in range(NUM_EDGES):
        for j in range(len(skeletons)):
            edge = EDGES[i]
            if skeletons[j][edge[0], 2] < visual_thresh or skeletons[j][edge[
                    1], 2] < visual_thresh:
                continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1])**2 + (Y[0] - Y[1])**2)**0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length / 2), stickwidth),
                                       int(angle), 0, 360, 1)
            if ids is None:
                color = colors[i] if color_set is None else colors[color_set[j]
                                                                   %
                                                                   len(colors)]
            else:
                color = get_color(ids[j])
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    if returnimg:
        return canvas
    save_name = os.path.join(
        save_dir, os.path.splitext(os.path.basename(imgfile))[0] + '_vis.jpg')
    plt.imsave(save_name, canvas[:, :, ::-1])
    print("keypoint visualize image saved to: " + save_name)
    plt.close()




def crop_image_with_det(batch_input, det_res, thresh=0.3):
    boxes = det_res['boxes']
    score = det_res['boxes'][:, 1]
    boxes_num = det_res['boxes_num']
    start_idx = 0
    crop_res, new_bboxes, ori_bboxes = [], [], []
    for b_id, input in enumerate(batch_input):
        boxes_num_i = boxes_num[b_id]
        if boxes_num_i == 0:
            continue
        boxes_i = boxes[start_idx:start_idx + boxes_num_i, :]
        score_i = score[start_idx:start_idx + boxes_num_i]
        res, res_nex_box, res_ori_box = [],[],[]
        for box, s in zip(boxes_i, score_i):
            if s > thresh:
                crop_image, new_box, ori_box = expand_crop(input, box)
                if crop_image is not None:
                    res.append(crop_image)
                    res_nex_box.append(new_box)
                    res_ori_box.append(ori_box)
        crop_res.append(res)
        new_bboxes.append(res_nex_box)
        ori_bboxes.append(res_ori_box)
        start_idx += boxes_num_i
    return crop_res, new_bboxes, ori_bboxes


def visualize_image( im_files, images, output_dir, det_res, kpt_res):
    start_idx, boxes_num_i = 0, 0

    for i, (im_file, im) in enumerate(zip(im_files, images)):
        if det_res is not None:
            det_res_i = {}
            boxes_num_i = det_res['boxes_num'][i]



        if kpt_res is not None:
            kpt_res_i = {}
            kpt_res_i['keypoint'] = [ kpt_res['keypoint'][0][start_idx:start_idx + boxes_num_i, :, :],
                                      kpt_res['keypoint'][1][start_idx:start_idx + boxes_num_i, :]
                                    ]
            kpt_res_i['bbox'] = kpt_res['bbox'][start_idx:start_idx + boxes_num_i, :]

            im = visualize_pose(
                im,
                kpt_res_i,
                visual_thresh=0.5,
                returnimg=True)

        start_idx += boxes_num_i

        img_name = os.path.split(im_file)[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, im)
        print("save result to: " + out_path)
