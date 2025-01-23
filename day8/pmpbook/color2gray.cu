#include <cuda_runtime.h>
#include <iostream>

__global__ void color2graykernel(const float* R, const float*G,const float*B,float *O,const int n){
    // assume the matrix is nxn;

    int i = blockIdx.x * blockDim.x + threadIdx.x; // so this will be for collumns
    int j = blockIdx.y * blockDim.y + threadIdx.y; // this will be for rows


    if( i<n && j<n){
        int idx = j* n + i;
        O[idx] = 0.299f * R[idx] + 0.587f * G[idx] + 0.114f * B[idx];
    }
}

float* color2gray(const float *R,const float*G,const float*B,const int n){
    float * d_r, *d_g, *d_b, *d_o;
    int size = n * n * sizeof(float);
    int Threads = 32;
    dim3 gridDim(ceil(n/Threads),ceil(n/Threads),1);
    dim3 BlockDim(Threads,Threads,1);

    cudaMalloc((void**)&d_r, size);
    cudaMalloc((void**)&d_g, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_o, size);

    cudaMemcpy(d_r,R,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_g,G,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,size,cudaMemcpyHostToDevice);

    color2graykernel<<<gridDim,BlockDim>>>(d_r,d_g,d_b,d_o,n);

    float *O = (float*)malloc(size);
    cudaMemcpy(O,d_o,size,cudaMemcpyDeviceToHost);

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_o);

    return O;
}