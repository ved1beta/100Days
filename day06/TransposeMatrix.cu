#include <iostream>
#include <cuda_runtime.h>

__global__ void transposeKernel(int *A, int *B)
{
    const int idx = threadIdx.x + threadIdx.y * blockDim.x;
    // threadIDx.x -> id of the row
    // threadIdx.y -> id of the collumn
    // BlockDim.x -> the size of the Dimension of the row
    // So we will get the idx to be on the element in the flattned matrix

    //  1  2  3    1  2  5
    //  2  3  4 -> 2  3  2
    //  5  2  1    3  4  1
    const int outidx = threadIdx.y + threadIdx.x * blockDim.y;
    B[outidx] = A[idx];
}

int main()
{
    int rows = 3;
    int cols = 3;
    int sizeMatrix = rows * cols;
    int *Matrix = (int *)malloc(sizeof(int) * cols * rows);
    for (int i = 0; i < sizeMatrix; i++)
    {
        Matrix[i] = i;
    }
    for (int i = 0; i < sizeMatrix; i++)
    {
        std::cout << Matrix[i] << " ";
        if (i % cols == cols - 1)
            std::cout << std::endl;
    }

    int *MatrixD, *MatrixOut;
    cudaMalloc((void **)&MatrixD, sizeMatrix * sizeof(int));
    cudaMalloc((void **)&MatrixOut, sizeMatrix * sizeof(int));
    cudaMemcpy(MatrixD, Matrix, sizeMatrix * sizeof(int), cudaMemcpyHostToDevice);

    dim3 numThreadsPerBlock(rows, cols);

    cudaFuncSetAttribute(
        transposeKernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        20 // Use 20% of combined L1/Shared Memory for Shared Memory
    ); 
    transposeKernel<<<1, numThreadsPerBlock>>>(MatrixD, MatrixOut);

    cudaMemcpy(Matrix, MatrixOut, sizeMatrix * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "\nTransposed\n";
    for (int i = 0; i < sizeMatrix; i++)
    {
        std::cout << Matrix[i] << " ";
        if (i % rows == rows - 1)
            std::cout << std::endl;
    }

    cudaFree(MatrixD);
    cudaFree(MatrixOut);
    free(Matrix);

    return 0;
}