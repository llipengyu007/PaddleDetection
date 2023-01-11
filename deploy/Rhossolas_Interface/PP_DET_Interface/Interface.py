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

    print(paddle.__version__)
    # image = '/Users/lipengyu/Downloads/bad_case/img4/2_8.png'
    image = './test_img/Image_20221229141402.png'
    image_deepcopy = './test_img/3e28e4af45765b0001d7a817.png'
    repeat = 1


    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    FLAGS.device = 'GPU'
    FLAGS.run_mode = 'trt_fp16'
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"
    #model_dir = '/Users/lipengyu/.cache/paddle/infer_weights/mot_ppyoloe_l_36e_pipeline'
    model_dir = './mot_ppyoloe_l_36e_pipeline'
    print('FLAGS.device is {}'.format(FLAGS.device))
    print('FLAGS.run_mode is {}'.format(FLAGS.run_mode))


    #initialization
    det_predictor = Detector(
        model_dir, FLAGS.device, FLAGS.run_mode, FLAGS.batch_size,
        FLAGS.trt_min_shape, FLAGS.trt_max_shape, FLAGS.trt_opt_shape,
        FLAGS.trt_calib_mode, FLAGS.cpu_threads, FLAGS.enable_mkldnn)



    # standard process start
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
    # standard process end

    '''
    The following code just for test the deep copy and process, there is no need in the deploy
    # process test start
    print('test process...')
    for i in range(repeat):
        inputs_test_process = det_predictor.preprocess(image)
        result = det_predictor.predict()
        result = det_predictor.postprocess(inputs_test_process, result)
        result = [result]
        det_res = det_predictor.merge_batch_result(result)

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
    # process test  end

    # deep copy test start
    det_predictor_deepcopy = paddle.clone(det_predictor)
    image_deepcopy = [decode_image(image_deepcopy, {})[0]]  # image for deep copy test
    print('test deepcopy')
    for i in range(repeat):
        # image processing and model inference
        det_res = det_predictor_deepcopy.predict_image(image_deepcopy, visual=False) #visualize is not support in this interface
        # post process for NMS
        det_res = det_predictor_deepcopy.filter_box(det_res,
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

    print('test memory...')
    for i in range(repeat):
        #inputs = det_predictor.preprocess(image)
        result = det_predictor.predict()
        result = det_predictor.postprocess(inputs_test_process, result)
        result = [result]
        det_res = det_predictor.merge_batch_result(result)

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
    # test deepcopy code process end
    '''

    print('complete')