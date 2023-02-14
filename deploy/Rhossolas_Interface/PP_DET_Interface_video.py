import os

import time
import paddle


import cv2
from PP_DET_Interface.Interface_utils import argsparser, decode_image
from deploy.Rhossolas_Interface.python.infer import Detector

if __name__ == '__main__':
    video = '/Users/lipengyu/Downloads/bad_case/tmp/video/1.avi'
    visual = False
    visual_output = '/Users/lipengyu/Downloads/bad_case/tmp/output/tt'
    if os.path.exists(visual_output) == False:
        os.system('mkdir -p {}'.format(visual_output))

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

    # initialization
    det_predictor = Detector(
        model_dir, FLAGS.device, FLAGS.run_mode, FLAGS.batch_size,
        FLAGS.trt_min_shape, FLAGS.trt_max_shape, FLAGS.trt_opt_shape,
        FLAGS.trt_calib_mode, FLAGS.cpu_threads, FLAGS.enable_mkldnn)

    cost = 0
    capture = cv2.VideoCapture(video)
    ind = 0
    while 1:
        ind += 1
        print(ind, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        ret, frame = capture.read()
        if ret == False or ind == 50:
            break
        image = [decode_image(frame, {})[0]]

        # image processing and model inference
        start = time.time()
        det_res = det_predictor.predict_image(image, visual=False)  # visualize is not support in this interface
        # post process for NMS
        det_res = det_predictor.filter_box(det_res,
            FLAGS.crop_thresh)
        cost += time.time() - start

        bboxes_num = det_res['boxes_num']
        bboxes = det_res['boxes']  # clsid, confidence, xmin, ymin, xmax, ymax
        assert len(bboxes) == bboxes_num
        for bbox in bboxes:
            print('class_id:{:d}, confidence:{:.4f}, left_top:[{:.2f},{:.2f}],'
                'right_bottom:[{:.2f},{:.2f}]'.format(
                int(bbox[0]), bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]))
            if visual:
                cv2.rectangle(frame, (int(bbox[2]), int(bbox[3])), (int(bbox[4]), int(bbox[5])), (0,0,0), 1)
                cv2.putText(frame, str(bbox[1]), (int(bbox[2]), int(bbox[3])), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 1)

        if visual:
            cv2.imwrite('{}/{}.jpg'.format(visual_output, ind), frame)

    print('complete, the cost per frame is {} s'.format(cost/ind))