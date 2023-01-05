import os

import cv2
import numpy as np
import time
import paddle
import sys

# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
print(parent_path)
exit(0)
sys.path.insert(0, parent_path)


from cfg_utils import argsparser, print_arguments, merge_cfg
from python.infer import Detector


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

if __name__ == '__main__':
    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    cfg = merge_cfg(FLAGS)
    print_arguments(cfg)

    image = '/Users/lipengyu/Downloads/bad_case/tmp/3e28e4af45765b0001d7a817.png'
    repeat = 1



    model_dir = '/Users/lipengyu/.cache/paddle/infer_weights/mot_ppyoloe_l_36e_pipeline'
    batch_size = 1
    crop_thresh = 0.5
    det_predictor = Detector(
        model_dir, FLAGS.device, FLAGS.run_mode, batch_size,
        FLAGS.trt_min_shape, FLAGS.trt_max_shape, FLAGS.trt_opt_shape,
        FLAGS.trt_calib_mode, FLAGS.cpu_threads, FLAGS.enable_mkldnn)

    start = time.time()

    image = [decode_image(image, {})[0]]
    for i in range(repeat):
        det_res = det_predictor.predict_image(image, visual=False)
        det_res = det_predictor.filter_box(det_res,
                                        crop_thresh)
    end = time.time()
    print('lipengyu cost:{}'.format((end - start) / repeat))

    bboxes_num = det_res['boxes_num']
    bboxes = det_res['boxes']  # clsid, confidence, xmin, ymin, xmax, ymax

    assert len(bboxes) == bboxes_num
    for bbox in bboxes:
        print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
              'right_bottom:[{:.2f},{:.2f}]'.format(
            int(bbox[0]), bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]))


    print('complete')