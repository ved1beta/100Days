#include <iostream>
#include <cuda_runtime.h>

// TODO : create template

struct Tensor {
    float *data;
    int rows;
    int cols;

    __device__ int size() const {
        return rows * cols;
    }

    __device__ float get(const int row, const int col = 0) {
        return data[row * cols + col];
    }

    __device__ void set(const int row, const int col = 0, const float value = 0) {
        data[row * cols + col] = value;
    }
};



__global__ void layerNorm(Tensor* A) {
    extern __shared__ float sharedMemory[];  

    float* mean = sharedMemory;
    float* variance = sharedMemory + A->rows;
    float* invstdev = sharedMemory + 2 * A->rows;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < A->rows) {
        mean[tid] = 0.0f;
        for (int j = 0; j < A->cols; j++) {
            mean[tid] += A->get(tid, j);
        }
        mean[tid] /= A->cols;
    }
    __syncthreads();

    if (tid < A->rows) {
        variance[tid] = 0.0f;
        for (int j = 0; j < A->cols; j++) {
            float diff = A->get(tid, j) - mean[tid];
            variance[tid] += diff * diff;
        }
        variance[tid] /= A->cols;
        invstdev[tid] = rsqrtf(variance[tid] + 1e-5f);
    }
    __syncthreads();

    if (tid < A->rows) {
        for (int j = 0; j < A->cols; j++) {
            float normalized = (A->get(tid, j) - mean[tid]) * invstdev[tid];
            A->set(tid, j, normalized);
        }
    }
}

int main() {
    const int rows = 1;
    const int cols = 3;
    const int tensorSize = rows * cols;
    const int size = rows * cols * sizeof(float);
    dim3 threadsPerBlock(16*16);
    dim3 numBlocks((rows + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemorySize = 3 * rows * sizeof(float);

    // Shared memory for mean, variance, invstdev
    // we need 3 different arrays of rows * size(float)

    // host data 
    float h_data[tensorSize] = {
        5.0f, 1.5f, 2.0f,
    };

    // now we allocate the data on the device and copy it 
    float* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    
    // now lets allocate the tensor on the device
    // we create our tensor but we use teh device data as the data
    // and we only copy the block

    Tensor h_tensor;
    h_tensor.data = d_data;
    h_tensor.rows = rows;
    h_tensor.cols = cols;

    Tensor* d_tensor;
    cudaMalloc(&d_tensor, sizeof(Tensor));
    cudaMemcpy(d_tensor, &h_tensor, sizeof(Tensor), cudaMemcpyHostToDevice);


    layerNorm<<<numBlocks, threadsPerBlock, sharedMemorySize>>>(d_tensor);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data,size, cudaMemcpyDeviceToHost);

    std::cout << "Normalized Matrix:" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << h_data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_tensor);

    return 0;
}