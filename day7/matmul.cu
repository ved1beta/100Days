#include <iostream>
#include <cuda_runtime.h>
#define BLOCK_SIZE 32

__global__ void matmulKernel(float *A, float *B, float *C, int dim)
{
    int i, j;       // i and j indexes
    float temp = 0; // temp value

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float ASharedT[BLOCK_SIZE][BLOCK_SIZE]; // we allocate memory for shared
    __shared__ float BSharedT[BLOCK_SIZE][BLOCK_SIZE]; // we allocate memory fro shared

    for (int tileNUM = 0; tileNUM < gridDim.x; tileNUM++)
    {
        j = tileNUM * BLOCK_SIZE + threadIdx.x;
        i = tileNUM * BLOCK_SIZE + threadIdx.y;

        ASharedT[threadIdx.y][threadIdx.x] = A[i * dim + j];
        BSharedT[threadIdx.y][threadIdx.x] = B[i * dim + j];

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            temp += ASharedT[threadIdx.y][k] * BSharedT[k][threadIdx.x];
        }

        __syncthreads();
    }
    C[row * dim + col] = temp;
}

int main()
{
    int N = 1024;
    float *Acpu, *Bcpu, *Ccpu;
    float *Agpu, *Bgpu, *Cgpu;

    Acpu = (float *)malloc(N * N * sizeof(float));
    Bcpu = (float *)malloc(N * N * sizeof(float));
    Ccpu = (float *)malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++)
    {
        Acpu[i] = sin(i);
        Bcpu[i] = cos(i);
    }

    size_t vectorSize = N * N * sizeof(float);

    cudaMalloc((void **)&Agpu, vectorSize);
    cudaMalloc((void **)&Bgpu, vectorSize);
    cudaMalloc((void **)&Cgpu, vectorSize);
    cudaMemcpy(Agpu, Acpu, vectorSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Bgpu, Bcpu, vectorSize, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(N / BLOCK_SIZE, N / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matmulKernel<<<gridDim, blockDim>>>(Agpu, Bgpu, Cgpu, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(Ccpu, Cgpu, vectorSize, cudaMemcpyDeviceToHost);

    printf("GPU time= %f ms\n", et);

    free(Acpu);
    free(Bcpu);
    free(Ccpu);
    cudaFree(Agpu);
    cudaFree(Bgpu);
    cudaFree(Cgpu);

    return 0;
}