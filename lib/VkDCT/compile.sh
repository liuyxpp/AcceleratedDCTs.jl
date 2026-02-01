#!/bin/bash

# 1. 下载头文件 (如果还没有)
# wget https://raw.githubusercontent.com/DTolm/VkFFT/master/vkFFT/VkFFT.h

# 2. 编译动态库
# -O3: 开启最高优化
# -shared -fPIC: 生成动态链接库
# -arch=sm_75: 针对 Turing 架构 (RTX 20xx) 优化
nvcc -O3 --shared -Xcompiler -fPIC -arch=sm_75 -o libvkfft_dct.so dct_shim.cu -I. -lcuda -lnvrtc