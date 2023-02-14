# 文件说明

- PP_DET_Interface: 主要针对PaddlePaddle的人类检测开发，应该对车辆等其他检测也是可以通过更换模型和配置文件更新（暂未测试）
  - Interface_utils.py: 一些简单的依赖额外函数。主要为：1. 清晰展示读图过程的decode_image函数；2. 配置默认参数的函数
- python文件夹:依赖包。需要导入，但是可以不看不关注。里面可能存在一些融于项没有删除，暂时不做处理
- PP_DET_Interface.py 和 PP_DET_Interface_video.py：检测的接口，里面实现了main函数，可以参考阅读。整体接口十分简单，容易封装。一个是针对图片，另一个是视频


# 这个接口需要依赖PaddlePaddle的相关组件，请参考如下链接安装：

- https://www.yuque.com/g/rhossolas/yb85sz/lasv9vfzorn65cer/collaborator/join?token=xtUM7iD4FBo4AmjC&goto=%2Frhossolas%2Fyb85sz%2Flasv9vfzorn65cer%2Fedit# 邀请你共同编辑文档《PaddlePaddle安装和使用tips》)    
- 仅仅需要安装PaddlePaddle就可以，不需要安装PaddleDetection
- 注意环境的匹配

# 强烈建议使用tensorrt进行加速，所以需要安装tensorrt包

- 可以直接通过 export LD_LIBRARY_PATH="XXXX:LD_LIBRARY_PATH" 进行安装生效    
- 注意tensorrt的版本，cudnn的版本，cuda的版本，以及nvidia-driver的版本需要匹配
- tensorRT下载地址为：https://developer.nvidia.com/nvidia-tensorrt-8x-download
- tensorRT安装介绍：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar

# 在本项目中，如果希望生效tensorrt加速，可以配置--run_mode进行:
- 本加速配置既可以在代码调用作为入参，也可以直接在代码中配置，详情参看Interface.py  
- run_mode 为paddle，普通非加速模式
- run_mode 为 trt_fp32/trt_fp16/trt_int8，tensorrt不同精度的加速模式。在有限的5张bad case上测试，fp32和fp16没有发现损失， int8感觉性价比不高，所以没有测试。
- 建议在满足速度的情况下，尽可能使用更高的精度
- 使用tensorrt的缺点为初始化的时候会比较慢

# 效率测试
  ## PP_DET_Interface模型的效率测试 
- 同一张图片测试1000次，模型为 [mot_ppyoloe_l_36e_pipeline][https://bj.bcebos.com/v1/paddledet/models/pipeline/mot_ppyoloe_l_36e_pipeline.zip]
- 测试环境
  - PaddlePaddle：2.4.1
  - tensorrt：8.4.1.5
  - cuda：11.6
  - 显卡：NVIDIA Corporation Device 2208 (rev a1)
  - 操作系统：Linux version 5.4.0-124-generic (buildd@lcy02-amd64-089) (gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.1)) #140-Ubuntu SMP Thu Aug 4 02:23:37 UTC 2022
- 测试结果（忽略读图IO时间）：
  - run_mode=paddle: 50.3ms
  - run_mode=trt_fp32: 17.4ms
  - run_mode=trt_fp16: 12.4ms
  - run_mode=trt_int8: 12.9ms

# 注意事项
- 本代码中的detector类会使用类似于handler一类的东西将input图片缓存进去，然后在用缓存的东西进行前向。所以当多路并发而公用一个detector的对象的时候会存在input串流的情况。如果要多路并发，建议每一路单独建立一个detector或者通过batch size进行分发。
- 在新建detector的如果直接init比较慢，可以考虑使用paddle.clone(XXXX)的形式

# 跟新说明
- 20230214： 重新组织文件格式，将python文件包放到接口依赖包外面
- 新增代码支持对于Detector类的deepcopy过程。但是要求paddlepaddle版本大于2.4.0