#include <cuda_runtime.h>

__global__ void matrixmulkernel(float *M,float *N,float *P,int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < width && col <width){
        float Pvalue = 0;
        for(int k = 0; k<width;++k){
            Pvalue += M[row * width + k] * N[k*width + col];
        }
        P[row * width + col] = Pvalue;
    }
}