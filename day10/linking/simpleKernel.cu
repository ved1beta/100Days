#include <stdio.h>
#include <cuda_runtime.h>
#include "ATen/ATen.h"

template <typename T>
__global__ void simpleKernel(T* A) {
    A[threadIdx.x] += 100;
}

void cuda_simpleKernel(float *A ) {
    dim3 blocks(1);
    simpleKernel<<<blocks, 32>>>(A);
}