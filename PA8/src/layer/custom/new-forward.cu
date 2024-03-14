#include "gpu-new-forward.h"
#include <cmath>
#include <iostream>

#define TW 16
#define ceil_div(X, Y) (X % Y == 0 ? X / Y : X / Y + 1)

__global__ void conv_forward_kernel(float *y, const float * __restrict__ x, const float * __restrict__ k,
                                    const int B, const int M, const int C,
                                    const int H, const int W, const int K) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire
    mini-batch The goal here is to be correct AND fast. Function paramter
    definitions: y - output x - input k - kernel B - batch_size (number of
    images in x) M - number of output feature maps C - number of input feature
    maps H - input height dimension W - input width dimension K - kernel height
    and width (K x K)
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    (void)H_out; // silence declared but never referenced warning. remove this
                 // line when you start working
    (void)W_out; // silence declared but never referenced warning. remove this
                 // line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to
    // use them, or create your own. An example use of these macros: float a =
    // y4d(0,0,0,0) y4d(0,0,0,0) = a

#define y4d(i3, i2, i1, i0)                                                    \
    y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0)                                                    \
    x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0)                                                    \
    k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    // Insert your GPU convolution kernel code here

    __shared__ float in_tile[TW][TW];
    __shared__ float k_tile[TW][TW];

    // iterate the tiles through the input and kernel arrays.
    int unrolled_kernel_width = C * K * K;
    int k_tile_thread_y = blockIdx.y * TW + threadIdx.y;
    int unrolled_tile_thread_x = blockIdx.z * TW + threadIdx.x;
    float accumulator = 0.0;
    int num_tiles = ceil_div(unrolled_kernel_width, TW);
    for (int i = 0; i < num_tiles; ++i) {
        // Copy this thread's appropriate coordinate into k_tile
        int k_tile_thread_x = i * TW + threadIdx.x;
        if (k_tile_thread_x < unrolled_kernel_width && k_tile_thread_y < M) {
            k_tile[threadIdx.y][threadIdx.x] =
                k[k_tile_thread_y * unrolled_kernel_width + k_tile_thread_x];
        } else {
            k_tile[threadIdx.y][threadIdx.x] = 0;
        }

        // Do the same for in_tile, though this is more complicated...
        int unrolled_tile_thread_y = i * TW + threadIdx.y;
        if (unrolled_tile_thread_x < W_out * H_out &&
            unrolled_tile_thread_y < unrolled_kernel_width) {
            int in_y = (unrolled_tile_thread_y % (K * K)) / K +
                       unrolled_tile_thread_x / W_out;
            int in_x = (unrolled_tile_thread_y % (K * K)) % K +
                       unrolled_tile_thread_x % W_out;
            int channel_idx = unrolled_tile_thread_y / (K*K);
            in_tile[threadIdx.y][threadIdx.x] = x4d(blockIdx.x, channel_idx, in_y, in_x);
        } else {
            in_tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // soom
        for(int j = 0; j < TW; ++j) {
            accumulator += k_tile[threadIdx.y][j] * in_tile[j][threadIdx.x];
        }

        __syncthreads();

    }

    if (k_tile_thread_y < M && unrolled_tile_thread_x < H_out * W_out) {
        y[(blockIdx.x * M + k_tile_thread_y) * H_out * W_out + unrolled_tile_thread_x] = accumulator;
    }

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

    // We pass double pointers for you to initialize the relevant device
    // pointers,
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
    k_sz = sizeof(float) * M * C * K * K;
    cudaMalloc(device_x_ptr, x_sz);
    cudaMalloc(device_y_ptr, y_sz);
    cudaMalloc(device_k_ptr, k_sz);

    cudaMemcpy(*device_x_ptr, host_x, x_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_k_ptr, host_k, k_sz, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(
    float *device_y, const float *device_x, const float *device_k, const int B,
    const int M, const int C, const int H, const int W, const int K) {
    // Set the kernel dimensions and call the kernel
    dim3 blockDim{TW, TW};
    const int conv_path_width = W - K + 1;
    const int conv_path_height = H - K + 1;
    const unsigned int unrolled_width = conv_path_width * conv_path_height;
    const unsigned int unrolled_height = M;
    dim3 gridDim{(unsigned int)B, ceil_div(unrolled_height, TW),
                 ceil_div(unrolled_width, TW)};
    conv_forward_kernel<<<gridDim, blockDim>>>(device_y, device_x, device_k, B,
                                               M, C, H, W, K);
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

        std::cout << "Device " << dev << " name: " << deviceProp.name
                  << std::endl;
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
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0]
                  << " x, " << deviceProp.maxGridSize[1] << " y, "
                  << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}

__global__ void fc_forward_kernel(float *device_y, const float *device_x,
                                  const float *device_w, const float *device_b,
                                  const int NUM, const int DIM_in,
                                  const int DIM_out) {
    // Extra credit
    /*
    Modify this function to implement a fully-connected layer.
    The goal is to be correct AND fast, you should consider using shared memory
    NUM: number of images
    device_x: input matrix
    device_w: weight matrix
    device_y: output matrix
    device_b: bias vector
    DIM_in: input vector size
    DIM_out: output vector size

    HINT: the functionality of the linear layer is equivalent to the forward
    pass implemented in ./src/layer/fully_connected.cc
    */
}

__host__ void GPUInterface::fc_forward_gpu_prolog(
    const float *host_y, const float *host_x, const float *host_w,
    const float *host_b, float **device_y_ptr, float **device_x_ptr,
    float **device_w_ptr, float **device_b_ptr, const int NUM, const int DIM_in,
    const int DIM_out) {
    // Extra credit
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device
    // pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
}

__host__ void GPUInterface::fc_forward_gpu(
    float *device_y, const float *device_x, const float *device_w,
    float *device_b, const int NUM, const int DIM_in, const int DIM_out) {
    // Extra credit
    // Set the kernel dimensions and call the kernel
}

__host__ void GPUInterface::fc_forward_gpu_epilog(
    float *host_y, float *device_y, float *device_x, float *device_w,
    float *device_b, const int NUM, const int DIM_in, const int DIM_out) {
    // Extra credit
    // Copy the output back to host

    // Free device memory
}
