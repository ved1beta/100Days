#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 3;  // Change to 1024 for performance testing
    float *A, *B, *C;

    // Allocate and initialize host memory
    A = (float*)malloc(N * N * sizeof(float));
    B = (float*)malloc(N * N * sizeof(float));
    C = (float*)malloc(N * N * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
        }
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * N * sizeof(float));
    cudaMalloc(&d_b, N * N * sizeof(float));
    cudaMalloc(&d_c, N * N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Configure kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                 (N + blockSize.y - 1) / blockSize.y);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Execute kernel
    cudaEventRecord(start);
    matmul<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Get elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back
    cudaMemcpy(C, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrices for small N
    if (N <= 10) {
        std::cout << "Matrix A:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << A[i * N + j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\nMatrix B:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << B[i * N + j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\nResult Matrix C:\n";
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << C[i * N + j] << " ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "\nCUDA Execution Time: " << milliseconds << " ms\n";

    // Cleanup
    free(A);
    free(B);
    free(C);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}