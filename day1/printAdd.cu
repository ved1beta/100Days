#include <iostream>
#include <cuda_runtime.h>

// Kernel to print threadIdx.x
__global__ void printThreadIdx(int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) { // Ensure the thread is within bounds
        printf("Block: %d, Thread: %d, Global Index: %d\n", blockIdx.x, threadIdx.x, idx);
    }
}

int main() {
    const int N = 1024; // Number of elements
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    printThreadIdx<<<blocksPerGrid, threadsPerBlock>>>(N);

    // Wait for the device to finish
    cudaDeviceSynchronize();

    return 0;
}
