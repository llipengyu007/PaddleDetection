import time
import paddle

from Interface_utils import argsparser, decode_image
from Interface_utils import crop_image_with_det, visualize_image


from python.infer import Detector
from python.keypoint_infer import KeyPointDetector

if __name__ == '__main__':

    print(paddle.__version__)
    image_path_list = ['/Users/lipengyu/Downloads/bad_case/tmp_2/42632f59dac8730001b940a8.png',
                       '/Users/lipengyu/Downloads/bad_case/tmp_2/528dc29fdac8730001b940c4.png']
    # image = './test_img/Image_20221229141402.png'
    # image_deepcopy = './test_img/3e28e4af45765b0001d7a817.png'
    repeat = 3
    visualize = True
    output_dir = '/Users/lipengyu/Downloads/bad_case/tmp_2/outttt'

    #assert repeat==1 or visualize==False

    paddle.enable_static()

    # parse params from command
    parser = argsparser()
    FLAGS = parser.parse_args()
    FLAGS.device = FLAGS.device.upper()
    FLAGS.device = 'CPU'
    #FLAGS.run_mode = 'trt_fp16'
    assert FLAGS.device in ['CPU', 'GPU', 'XPU'
                            ], "device should be CPU, GPU or XPU"

    print('FLAGS.device is {}'.format(FLAGS.device))
    print('FLAGS.run_mode is {}'.format(FLAGS.run_mode))


    #initialization Det
    det_model_dir = '/Users/lipengyu/.cache/paddle/infer_weights/mot_ppyoloe_l_36e_pipeline'
    # model_dir = './mot_ppyoloe_l_36e_pipeline'
    # FLAGS.batch_size = len(image_path_list) # example coe for batch is more than 1
    det_predictor = Detector(
        det_model_dir, FLAGS.device, FLAGS.run_mode, FLAGS.batch_size,
        FLAGS.trt_min_shape, FLAGS.trt_max_shape, FLAGS.trt_opt_shape,
        FLAGS.trt_calib_mode, FLAGS.cpu_threads, FLAGS.enable_mkldnn)

    #initialization KPY
    kpt_model_dir = '/Users/lipengyu/.cache/paddle/infer_weights/dark_hrnet_w32_256x192'
    kpt_batch_size = 8

    kpt_predictor = KeyPointDetector(
        kpt_model_dir,
        FLAGS.device,
        FLAGS.run_mode,
        kpt_batch_size,
        FLAGS.trt_min_shape,
        FLAGS.trt_max_shape,
        FLAGS.trt_opt_shape,
        FLAGS.trt_calib_mode,
        FLAGS.cpu_threads,
        FLAGS.enable_mkldnn,
        use_dark=False)
    #kpt_predictor = paddle.clone(kpt_predictor) # deepcopy support

    # standard process start

    cost_det, cost_kpt, cost_vis = 0, 0, 0 #param for speed statistics
    num_det, num_kpt, num_vis = 0, 0, 0 #param for speed statistics
    #batch_input = [decode_image(f, {})[0] for f in batch_file]
    for i in range(repeat):
        for image_path in image_path_list:
            # loading image
            image = [decode_image(image_path, {})[0]]

            # image = [decode_image(f, {})[0] for f in image_path_list] # example coe for batch is more than 1

            start = time.time()
            # image processing and Det model inference
            det_res = det_predictor.predict_image(image, visual=False) #visualize is not support in this interface
            # post process for NMS
            det_res = det_predictor.filter_box(det_res,
                                            FLAGS.crop_thresh)
            cost_det += time.time() - start
            num_det += 1

            start = time.time()
            # image Croping
            crop_inputs, new_bboxes, ori_bboxes = crop_image_with_det(image, det_res)
            # image processing and KPY model inference

            kpt_res = kpt_predictor.predict_batch_image(crop_inputs, new_bboxes, ori_bboxes, visual=False) #visualize is not support in this interface
            cost_kpt += time.time() - start
            num_kpt += len(kpt_res['bbox'])

            # shape of skeleton is [skeleton number, key point number, key point location]
            # key point location == x, y, conf_for_loc
            print('shape of skeleton: {}'.format(kpt_res['keypoint'][0].shape))
            print('shape of confidence score of the skeleton: {}'.format(kpt_res['keypoint'][1].shape))

            if visualize:
                start = time.time()
                visualize_image([image_path], image, output_dir, det_res, kpt_res)
                # visualize_image(image_path_list, image, output_dir, det_res, kpt_res) # example coe for batch is more than 1
                cost_vis = time.time() - start
                num_vis += int(det_res['boxes_num'])



    print('repeat num: {}, imag_num:{}, bbox num: {}'.format(repeat, num_det,num_kpt))
    print('det cost: {:.4f}'.format(cost_det / num_det))
    print('kpt cost: {:.4f}'.format(cost_kpt / num_kpt))
    print('vis cost: {:.4f}'.format(cost_vis / num_vis))


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