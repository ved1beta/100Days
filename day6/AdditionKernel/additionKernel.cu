#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename T>
__global__ void addKernel(T* input, int arraySize) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arraySize) {
        input[idx] += 10;
    }
}

void addition(torch::Tensor& input, int arraySize) {
    int threads_per_block = 256;
    int blocks = (arraySize + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "arrayAddition", [&]() {
        addKernel<<<blocks, threads_per_block>>>(input.data_ptr<scalar_t>(), arraySize);
    });
    cudaDeviceSynchronize();
    
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
    }
}   