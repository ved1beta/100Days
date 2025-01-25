#include <iostream>
#include <cuda_runtime.h>

__global__ void conv1D(float *X, float *K, float *Y, int input_size, int kernel_size)
{

    extern __shared__ float shared[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int radius = kernel_size / 2;

    int sharedIdx = threadIdx.x + radius; // the main element from the conv
    // index will start from the radius so that we have left 2 more behind use 
    /// SO we load in the share memory all the elements our filter will work on the block
    if (threadIdx.x < blockDim.x - radius)
    {
        int left = i - radius;
        int right = i + blockDim.x;

        shared[threadIdx.x] = (left >= 0) ? X[left] : 0.0f; 
        shared[sharedIdx + blockDim.x] = (right < input_size) ? X[right] : 0.0f;
    }

    __syncthreads();

    float sum = 0.0;
    for (int j = -radius; j <= radius; j++)
    {
        sum += shared[sharedIdx + j] * K[radius + j];
        // we iterate from -2 to 2 . so we have -2 -1 0 1 2. Which is normal
        // So we have this:
    }

    if (i < input_size)
    {
        Y[i] = sum;
    }
}

int main()
{
    int N = 1024;                                   // size of the vector
    int BlockSize = 256;                            // size of the block we use
    int GridSize = (N + BlockSize - 1) / BlockSize; // size of the grid we use. Also ceil function

    int KernelSize = 5;
    float Kernel[KernelSize] = {1.0f, 2.0f, 1.0f, 1.0f, -2.0f};
    int radius = KernelSize / 2;
    int SharedMemory = (BlockSize + 2 * radius) * sizeof(float);

    float *Xcpu, *Ycpu;
    float *Xgpu, *Ygpu, *Kgpu;

    Xcpu = (float *)malloc(N * sizeof(float));
    Ycpu = (float *)malloc(N * sizeof(float));
    // we already have declared our kernel;

    for (int i = 0; i < N; i++)
    {
        Xcpu[i] = 1;
    }

    // now lets launch this data in the air baby
    cudaMalloc((void **)&Xgpu, N * sizeof(float));
    cudaMalloc((void **)&Ygpu, N * sizeof(float));
    cudaMalloc((void **)&Kgpu, KernelSize * sizeof(float));
    cudaMemcpy(Xgpu, Xcpu, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Kgpu, Kernel, KernelSize * sizeof(float), cudaMemcpyHostToDevice);

    conv1D<<<GridSize, BlockSize, SharedMemory>>>(Xgpu, Kgpu, Ygpu, N, KernelSize);

    cudaMemcpy(Ycpu, Ygpu, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "First 10 elements " << std::endl;
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << Xcpu[i] << " ";
    }

    std::cout << "\nFirst 10 elements after the convolution op" << std::endl;
    for (size_t i = 0; i < 10; i++)
    {
        std::cout << Ycpu[i] << " ";
    }

    free(Xcpu);
    free(Ycpu);
    cudaFree(Xgpu);
    cudaFree(Ygpu);
    cudaFree(Kgpu);

    return 0;
}
