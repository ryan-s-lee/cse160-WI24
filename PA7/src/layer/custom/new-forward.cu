#include "gpu-new-forward.h"
// #include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <iostream>

#define TILE_WIDTH 16
#define ceil_div(X,Y) X % Y == 0 ? X / Y : X / Y + 1
__global__ void conv_forward_kernel(float *y, const float *x, const float *k,
                                    const int B, const int M, const int C,
                                    const int H, const int W, const int K) {

  /*
  Modify this function to implement the forward pass described in Chapter 16.
  We have added an additional dimension to the tensors to support an entire
  mini-batch The goal here is to be correct AND fast. We have some nice #defs
  for you below to simplify indexing. Feel free to use them, or create your own.
  */

  const int H_out = H - K + 1;
  const int W_out = W - K + 1;
  (void)H_out; // silence declared but never referenced warning. remove this
               // line when you start working
  (void)W_out; // silence declared but never referenced warning. remove this
               // line when you start working

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0)                                                    \
  y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0)                                                    \
  x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0)                                                    \
  k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

  // map threadidx to b, m, h_out, w_out
  int b, m, h_out, w_out, c, p, q;
  b = blockIdx.x;
  h_out = blockIdx.y * TILE_WIDTH + threadIdx.y;
  w_out = blockIdx.x * TILE_WIDTH + threadIdx.x;
  if (h_out > H_out || w_out > W_out) return;
  m = blockIdx.z;
  float accum = 0.0;
  for (c = 0; c < C; ++c) {
    for (p = 0; p < K; ++p) {
      int h = h_out + p;
      for (q = 0; q < K; ++q) {
        int w = w_out + q;
        accum += x4d(b, c, h, w) * k4d(m, c, h, w);
      }
    }
  }
  y4d(b, m, h_out, w_out) = accum;

#undef y4d
#undef x4d
#undef k4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_y, const float *host_x, const float *host_k,
    float **device_y_ptr, float **device_x_ptr, float **device_k_ptr,
    const int B, const int M, const int C, const int H, const int W,
    const int K) {
  // Allocate memory and copy over the relevant data structures to the GPU

  // We pass double pointers for you to initialize the relevant device pointers,
  //  which are passed to the other two functions.

  // Useful snippet for error checking
  // cudaError_t error = cudaGetLastError();
  // if(error != cudaSuccess)
  // {
  //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
  //     exit(-1);
  // }

  size_t x_sz, y_sz, k_sz;
  x_sz = sizeof(float) * B * H * W * C;
  const int H_out = H - (K - 1);
  const int W_out = W - (K - 1);
  y_sz = sizeof(float) * B * H_out * W_out * M;
  k_sz = y_sz * C * K * K;
  cudaMalloc(device_x_ptr, x_sz);
  cudaMalloc(device_y_ptr, y_sz);
  cudaMalloc(device_k_ptr, k_sz);
}

__host__ void GPUInterface::conv_forward_gpu(
    float *device_y, const float *device_x, const float *device_k, const int B,
    const int M, const int C, const int H, const int W, const int K) {
  // Set the kernel dimensions and call the kernel

  const int H_out = H - (K - 1);
  const int W_out = W - (K - 1);
  unsigned int W_grid, H_grid;
  H_grid = ceil_div(H_out, TILE_WIDTH);
  W_grid = ceil_div(W_out, TILE_WIDTH);
  dim3 gridDim{(unsigned int)M, H_grid * W_grid, (unsigned int)B};
  dim3 blockDim{TILE_WIDTH,TILE_WIDTH};
  conv_forward_kernel<<<gridDim,blockDim>>>(device_y, device_x, device_k, B, M, C, H, W, K);
}

__host__ void
GPUInterface::conv_forward_gpu_epilog(float *host_y, float *device_y,
                                      float *device_x, float *device_k,
                                      const int B, const int M, const int C,
                                      const int H, const int W, const int K) {
  // Copy the output back to host
  const int H_out = H - (K - 1);
  const int W_out = W - (K - 1);
  size_t y_sz = sizeof(float) * B * H_out * W_out * M;
  cudaMemcpy(host_y, device_y, y_sz, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(device_y);
  cudaFree(device_x);
  cudaFree(device_k);
}

__host__ void GPUInterface::get_device_properties() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);

  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
    std::cout << "Computational capabilities: " << deviceProp.major << "."
              << deviceProp.minor << std::endl;
    std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem
              << std::endl;
    std::cout << "Max Constant memory size: " << deviceProp.totalConstMem
              << std::endl;
    std::cout << "Max Shared memory size per block: "
              << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock
              << std::endl;
    std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0]
              << " x, " << deviceProp.maxThreadsDim[1] << " y, "
              << deviceProp.maxThreadsDim[2] << " z" << std::endl;
    std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
              << deviceProp.maxGridSize[1] << " y, "
              << deviceProp.maxGridSize[2] << " z" << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
  }
}
