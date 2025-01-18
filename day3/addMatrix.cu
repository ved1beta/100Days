#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

void printMatrix(const float *Matrix, const int size = 16) {
    int rootSize = sqrt(size);
    for (int i = 0; i < rootSize; i++) {
        for (int j = 0; j < rootSize; j++) {
            std::cout << Matrix[i * rootSize + j] << " ";
        }
        std::cout << "\n";
    }
}

__global__ void matrixAddCUDA(const float *Matrix_A, const float *Matrix_B, float *Matrix_C,
                              const int sizeX, const int sizeY) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < sizeY && col < sizeX) {
        Matrix_C[row * sizeX + col] = Matrix_A[row * sizeX + col] + Matrix_B[row * sizeX + col];
    }
}

void matrixAddCPU(const float *Matrix_A, const float *Matrix_B, float *Matrix_C, int sizeX, int sizeY) {
    for (int row = 0; row < sizeY; row++) {
        for (int col = 0; col < sizeX; col++) {
            Matrix_C[row * sizeX + col] = Matrix_A[row * sizeX + col] + Matrix_B[row * sizeX + col];
        }
    }
}

void compareExecutionTime(const float *Matrix_A, const float *Matrix_B, float *Matrix_C,
                          const int sizeX, const int sizeY) {
    const int matrixSize = sizeX * sizeY;
    const int matrixBytes = sizeof(float) * matrixSize;

    float *gpu_A, *gpu_B, *gpu_C;
    cudaMalloc((void **)&gpu_A, matrixBytes);
    cudaMalloc((void **)&gpu_B, matrixBytes);
    cudaMalloc((void **)&gpu_C, matrixBytes);

    cudaMemcpy(gpu_A, Matrix_A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_B, Matrix_B, matrixBytes, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((sizeX + BLOCK_SIZE - 1) / BLOCK_SIZE, (sizeY + BLOCK_SIZE - 1) / BLOCK_SIZE);

    auto startCPU = std::chrono::high_resolution_clock::now();
    matrixAddCPU(Matrix_A, Matrix_B, Matrix_C, sizeX, sizeY);
    auto endCPU = std::chrono::high_resolution_clock::now();

    auto startCUDA = std::chrono::high_resolution_clock::now();
    matrixAddCUDA<<<gridDim, blockDim>>>(gpu_A, gpu_B, gpu_C, sizeX, sizeY);
    cudaDeviceSynchronize();
    auto endCUDA = std::chrono::high_resolution_clock::now();

    cudaMemcpy(Matrix_C, gpu_C, matrixBytes, cudaMemcpyDeviceToHost);

    std::chrono::duration<double> cpuDuration = endCPU - startCPU;
    std::chrono::duration<double> cudaDuration = endCUDA - startCUDA;
    std::cout << "CPU Execution Time: " << cpuDuration.count() << " seconds\n";
    std::cout << "CUDA Execution Time: " << cudaDuration.count() << " seconds\n";

    cudaFree(gpu_A);
    cudaFree(gpu_B);
    cudaFree(gpu_C);
}

int main() {
    const int sizeX = 1024*16;
    const int sizeY = 1024*16;
    const int matrixSize = sizeX * sizeY;

    float *cpu_A = new float[matrixSize];
    float *cpu_B = new float[matrixSize];
    float *cpu_C = new float[matrixSize];

    for (int i = 0; i < matrixSize; i++) {  
        cpu_A[i] = 10.0f;
        cpu_B[i] = static_cast<float>(i);
    }

    compareExecutionTime(cpu_A, cpu_B, cpu_C, sizeX, sizeY);

    delete[] cpu_A;
    delete[] cpu_B;
    delete[] cpu_C;

    return 0;
}
