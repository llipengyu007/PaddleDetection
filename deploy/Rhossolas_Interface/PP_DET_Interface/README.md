# 这个接口需要依赖PaddlePaddle的相关组件，请参考如下链接安装：

. https://www.yuque.com/g/rhossolas/yb85sz/lasv9vfzorn65cer/collaborator/join?token=xtUM7iD4FBo4AmjC&goto=%2Frhossolas%2Fyb85sz%2Flasv9vfzorn65cer%2Fedit# 邀请你共同编辑文档《PaddlePaddle安装和使用tips》)    
. 仅仅需要安装PaddlePaddle就可以不需要安装PaddleDetection
. 注意环境的匹配

# 强烈建议使用tensorrt进行加速，所以需要安装tensorrt包

- 可以直接通过 export LIB_LABARY_PATH="XXXX:$LIB_LABARY_PATH" 进行安装生效    
- 注意tensorrt的版本，cudnn的版本，cuda的版本，以及nvidia-driver的版本需要匹配
- 在本项目中，如果希望生效tensorrt加速，可以配置--run_mode进行:
  - run_mode 为paddle，普通非加速模式
  - run_mode 为 trt_fp32/trt_fp16/trt_int8，tensorrt不同精度的加速模式。在有限的5张bad case上测试，fp16没有发现损失。
  - 建议在满足速度的情况下，尽可能使用更高的精度

# 文件说明

. python文件夹；依赖包。需要导入，但是可以不看不关注。里面可能存在一些融于项没有删除，暂时不做处理
. Interface.py：检测的接口，里面实现了main函数，可以参考阅读。整体接口十分简单，容易封装
. Interface_utils.py: 一些简单的依赖额外函数。主要为：1. 清晰展示读图过程的decode_image函数；2. 配置默认参数的函数
