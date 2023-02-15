from python.keypoint_preprocess import expand_crop
from python.visualize import visualize_pose
# from python.preprocess import decode_image

import os

import cv2
import numpy as np
import yaml
import ast
from argparse import ArgumentParser, RawDescriptionHelpFormatter


def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
    im_info['im_shape'] = np.array(im.shape[:2], dtype=np.float32)
    im_info['scale_factor'] = np.array([1., 1.], dtype=np.float32)
    return im, im_info


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='*', help="set configuration options")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)

        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=', 1)
            if '.' not in k:
                config[k] = yaml.load(v, Loader=yaml.Loader)
            else:
                keys = k.split('.')
                if keys[0] not in config:
                    config[keys[0]] = {}
                cur = config[keys[0]]
                for idx, key in enumerate(keys[1:]):
                    if idx == len(keys) - 2:
                        cur[key] = yaml.load(v, Loader=yaml.Loader)
                    else:
                        cur[key] = {}
                        cur = cur[key]
        return config

def argsparser():
    parser = ArgsParser()

    parser.add_argument(
        "--run_mode",
        type=str,
        default='paddle',
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU."
    )
    parser.add_argument(
        "--trt_calib_mode",
        type=bool,
        default=False,
        help="If the model is produced by TRT offline quantitative "
             "calibration, trt_calib_mode need to set True.")
    parser.add_argument(
        "--cpu_threads", type=int, default=1, help="Num of threads with CPU.")
    parser.add_argument(
        "--enable_mkldnn",
        type=ast.literal_eval,
        default=False,
        help="Whether use mkldnn with CPU.")
    parser.add_argument(
        "--trt_min_shape", type=int, default=1, help="min_shape for TensorRT.")
    parser.add_argument(
        "--trt_max_shape",
        type=int,
        default=1280,
        help="max_shape for TensorRT.")
    parser.add_argument(
        "--trt_opt_shape",
        type=int,
        default=640,
        help="opt_shape for TensorRT.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size.")
    parser.add_argument(
        "--crop_thresh",
        type=float,
        default=0.5,
        help="post process thre.")

    return parser

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
