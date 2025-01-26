#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void MatrixAdd_B(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        C[row * N + col] = A[row * N + col] + B[row * N + col];
    }
}

int main() {
    const int N = 4;  // Changed to smaller size for custom matrices
    float *A, *B, *C;

    // Custom matrix initialization
    A = (float *)malloc(N * N * sizeof(float));
    B = (float *)malloc(N * N * sizeof(float));
    C = (float *)malloc(N * N * sizeof(float));

    // First custom matrix (increasing values)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i * N + j + 1.0f;  // Values from 1 to 16
        }
    }

    // Second custom matrix (alternating pattern)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = (i + j) % 3 + 2.0f;  // Values alternating 2, 3, 4
        }
    }

    // Initialize result matrix to zero
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
        }
    }

    // CUDA device memory allocation
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * N * sizeof(float));
    cudaMalloc((void **)&d_b, N * N * sizeof(float));
    cudaMalloc((void **)&d_c, N * N * sizeof(float));

    // Copy host matrices to device
    cudaMemcpy(d_a, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Setup grid and block dimensions
    dim3 dimBlock(2, 2);  // Adjusted for smaller matrix
    dim3 dimGrid(ceil(N / 2.0f), ceil(N / 2.0f));

    // Launch kernel
    MatrixAdd_B<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, d_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print matrices
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", B[i * N + j]);
        }
        printf("\n");
    }

    printf("\nResultant Matrix C (A + B):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(A);
    free(B);
    free(C);

    return 0;
}