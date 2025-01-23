#include <iostream>
#include <cuda_runtime.h>

//// CHAPTER 2 DONE
__global__ void addkernel(float *a, float *b, float *c, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}


void vecAdd(float *A, float *B, float*C,int n){
    int size = n*sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n/256.0),1,1);
    dim3 dimBlock(256,1,1);
    addkernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, n);
    // launches a gri of 4 blocks with 256 threads per block

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);    
}