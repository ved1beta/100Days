#include <iostream>
#include <cuda_runtime.h>

__global__ void addition(float* A , float* B, float*C){
    int idx = threadIdx.x;
    C[idx]  = A[idx] + B[idx];
}

int main(){
    int N = 10;
    addition<<<1,N>>>(A,B,C); // simple addition kernle that will launch N threads
}

//////////////////////////////////////
// (x, y) is (x + y Dx);
// (x, y, z) is (x + y Dx + z Dx Dy)
// int i 
__global__ void MatAdd(float A[N][N], float B[N][N],
                       float C[N][N])
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    // Kernel invocation with one block of N * N * 1 threads
    int numBlocks = 1; // number of blocks
    dim3 threadsPerBlock(N, N); // Threads 
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}


/////////////////
// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N],
float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    ...
    // Kernel invocation
    dim3 threadsPerBlock(16, 16); // threadsPerBlock -> how many threads per blocl
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y); // Nu
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
    ...
}
////