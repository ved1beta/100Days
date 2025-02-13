#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

__global__ void LayerNorm(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    // Shared memory for the row data and intermediate calculations
    extern __shared__ float shared[];
    float* row_data = shared;
    
    // Load data into shared memory
    for (int i = tid; i < cols; i += blockDim.x) {
        row_data[i] = input[row * cols + i];
    }
    __syncthreads();
    
    // Calculate mean using parallel reduction
    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        sum += row_data[i];
    }
    
    // Parallel reduction in shared memory
    __shared__ float reduce_sum[32];  // Assuming max 32 warps
    int warp = tid / 32;
    int lane = tid % 32;
    
    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane == 0) {
        reduce_sum[warp] = sum;
    }
    __syncthreads();
    
    // First warp does final reduction
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? reduce_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            reduce_sum[0] = sum;
        }
    }
    __syncthreads();
    
    float mean = reduce_sum[0] / cols;
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_data[i] - mean;
        var_sum += diff * diff;
    }
    
    // Reduce variance similar to mean
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    
    if (lane == 0) {
        reduce_sum[warp] = var_sum;
    }
    __syncthreads();
    
    if (warp == 0) {
        var_sum = (lane < (blockDim.x + 31) / 32) ? reduce_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
        if (lane == 0) {
            reduce_sum[0] = var_sum;
        }
    }
    __syncthreads();
    
    float variance = reduce_sum[0] / cols;
    float stddev = sqrtf(variance + 1e-5f);
    
    // Normalize the data
    for (int i = tid; i < cols; i += blockDim.x) {
        output[row * cols + i] = (row_data[i] - mean) / stddev;
    }
}

int main() {
    const int rows = 10, cols = 10;
    float *A, *B;
    
    A = (float*)malloc(rows * cols * sizeof(float));
    B = (float*)malloc(rows * cols * sizeof(float));
    
    // Initialize input with random values
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    
    float *d_a, *d_b;
    cudaMalloc(&d_a, rows * cols * sizeof(float));
    cudaMalloc(&d_b, rows * cols * sizeof(float));
    
    cudaMemcpy(d_a, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with 1 block per row, using enough threads for the columns
    int threadsPerBlock = 256;
    dim3 gridDim(rows);
    dim3 blockDim(threadsPerBlock);
    size_t sharedMemSize = cols * sizeof(float) + 32 * sizeof(float); // row data + reduction space
    
    LayerNorm<<<gridDim, blockDim, sharedMemSize>>>(d_a, d_b, rows, cols);
    
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    cudaMemcpy(B, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Input:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", A[i * cols + j]);
        }
        printf("\n");
    }
    
    printf("\nLayer Normalized Output:\n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", B[i * cols + j]);
        }
        printf("\n");
    }
    
    // Verify the normalization
    printf("\nVerification (each row should have mean ≈ 0 and variance ≈ 1):\n");
    for (int i = 0; i < rows; i++) {
        float mean = 0.0f, var = 0.0f;
        for (int j = 0; j < cols; j++) {
            mean += B[i * cols + j];
        }
        mean /= cols;
        
        for (int j = 0; j < cols; j++) {
            float diff = B[i * cols + j] - mean;
            var += diff * diff;
        }
        var /= cols;
        printf("Row %d: mean = %.6f, variance = %.6f\n", i, mean, var);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    free(A);
    free(B);
    
    return 0;
}