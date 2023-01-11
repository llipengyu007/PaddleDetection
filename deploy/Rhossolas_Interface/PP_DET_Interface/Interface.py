import os

import time
import paddle
import sys

from Interface_utils import decode_image

'''
# add deploy path of PaddleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 3)))
print(parent_path)
sys.path.insert(0, parent_path)
parent_path = os.path.join(parent_path, 'pipeline')
print(parent_path)
sys.path.insert(0, parent_path)
'''

from Interface_utils import argsparser
from python.infer import Detector


if __name__ == '__main__':
    tag = {}
    tag[1] = 3
    tag[2] = 4
    while 1 in tag:
        print(1)

if __name__ == '__main__111':

    image = '/Users/lipengyu/Downloads/bad_case/img4/2_8.png'
    repeat = 1


    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"
    model_dir = '/Users/lipengyu/.cache/paddle/infer_weights/mot_ppyoloe_l_36e_pipeline'
    print('FLAGS.device is {}'.format(FLAGS.device))
    print('FLAGS.run_mode is {}'.format(FLAGS.run_mode))


    #initialization
    det_predictor = Detector(
        model_dir, FLAGS.device, FLAGS.run_mode, FLAGS.batch_size,
        FLAGS.trt_min_shape, FLAGS.trt_max_shape, FLAGS.trt_opt_shape,
        FLAGS.trt_calib_mode, FLAGS.cpu_threads, FLAGS.enable_mkldnn)

    start = time.time()

    image = [decode_image(image, {})[0]]
    for i in range(repeat):
        # image processing and model inference
        det_res = det_predictor.predict_image(image, visual=False) #visualize is not support in this interface
        # post process for NMS
        det_res = det_predictor.filter_box(det_res,
                                        FLAGS.crop_thresh)
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