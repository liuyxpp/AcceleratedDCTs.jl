// dct_shim.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

// 定义后端为 CUDA
#define VKFFT_BACKEND 1
#include "vkFFT.h"

// 上下文结构体：保存 VkFFT 的 App 实例，避免重复初始化
struct VkDCTContext {
    VkFFTApplication app;
    VkFFTConfiguration config;
    CUdevice device;
};

extern "C" {

/**
 * Create 3D DCT-I Plan
 * @param nx  X dimension
 * @param ny  Y dimension
 * @param nz  Z dimension
 * @param precision 0 for Float32, 1 for Float64
 * @return    Pointer to VkDCTContext (void*)
 */
void* create_dct3d_plan(uint64_t nx, uint64_t ny, uint64_t nz, int precision) {
    VkDCTContext* ctx = new VkDCTContext();
    if (!ctx) return nullptr;

    // 清零并配置
    ctx->app = {};
    ctx->config = {};

    // 1. 设置 3D 尺寸
    ctx->config.FFTdim = 3;
    ctx->config.size[0] = nx; // Fastest dim
    ctx->config.size[1] = ny;
    ctx->config.size[2] = nz; // Slowest dim

    // 2. Enable DCT-I
    // performDCT takes the DCT type (1-4). 1 for DCT-I.
    ctx->config.performDCT = 1;

    // 2b. Precision Configuration
    if (precision == 1) {
        ctx->config.doublePrecision = 1;
    }
    
    // 3. Normalization Settings
    // VkFFT divides by total element count on inverse transform by default.
    // Normalized to 0 for flexibility and high performance.
    ctx->config.normalize = 0; 

    // 4. 获取设备属性
    // 使用 Driver API 获取 CUdevice
    // 初始化 CUDA Driver API
    if (cuInit(0) != CUDA_SUCCESS) {
         delete ctx;
         return nullptr;
    }

    if (cuDeviceGet(&ctx->device, 0) != CUDA_SUCCESS) {
        delete ctx;
        return nullptr;
    }
    ctx->config.device = &ctx->device;
    ctx->config.num_streams = 1; // Default to 1 stream

    // 5. 初始化 (编译 Kernel)
    VkFFTResult res = initializeVkFFT(&ctx->app, ctx->config);
    
    if (res != VKFFT_SUCCESS) {
        printf("[VkDCT] Plan creation failed. Error code: %d\n", res);
        delete ctx;
        return nullptr;
    }

    return (void*)ctx;
}

/**
 * Execute Transform
 * @param plan_ptr  Pointer returned by create_dct3d_plan
 * @param buffer    Device memory pointer (void*) - types handled by internal config
 * @param stream    CUDA Stream
 * @param inverse   Direction (0: Forward, 1: Inverse)
 */
int exec_dct3d(void* plan_ptr, void* buffer, cudaStream_t stream, int inverse) {
    if (!plan_ptr) return -1;
    VkDCTContext* ctx = (VkDCTContext*)plan_ptr;

    // Update stream in the application configuration
    ctx->app.configuration.stream = &stream;

    VkFFTLaunchParams launchParams = {};
    launchParams.buffer = (void**)&buffer;       // Main buffer
    launchParams.inputBuffer = (void**)&buffer;  // Consistency
    launchParams.outputBuffer = (void**)&buffer; // Consistency
    
    // direction: -1 (Forward), 1 (Inverse)
    int dir = inverse ? 1 : -1;

    return (int)VkFFTAppend(&ctx->app, dir, &launchParams);
}

/**
 * 销毁计划，释放内存
 */
void destroy_dct3d_plan(void* plan_ptr) {
    if (plan_ptr) {
        VkDCTContext* ctx = (VkDCTContext*)plan_ptr;
        deleteVkFFT(&ctx->app);
        delete ctx;
    }
}

}