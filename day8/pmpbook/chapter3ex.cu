#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <algorithm>

#define CUDA_CHECK(err)                         \
    {                                           \
        cuda_assert((err), __FILE__, __LINE__); \
    }
inline void cuda_assert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " in " << file << ":" << line << std::endl;
        exit(1);
    }
}

__global__ void addKernelOne2One(const float *A, const float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[j * N + i] = A[j * N + i] + B[j * N + i];
}

__global__ void addKernelCol(const float *A, const float *B, float *C, const int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N)
    {
        for (int row = 0; row < N; ++row)
        {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

__global__ void addKernelRow(const float *A, const float *B, float *C, const int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N)
    {
        for (int col = 0; col < N; ++col)
        {
            C[row * N + col] = A[row * N + col] + B[row * N + col];
        }
    }
}

void benchmark_kernel(const char *name, void (*kernel)(const float *, const float *, float *, int),
                      const float *d_A, const float *d_B, float *d_C, int N, dim3 grid, dim3 block)
{
    const int runs = 5;
    float min_time = INFINITY;

    for (int i = 0; i < runs; ++i)
    {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        min_time = std::min(min_time, milliseconds);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::cout << "Kernel: " << name
              << " | Min time: " << min_time << " ms"
              << " | Grid: (" << grid.x << "," << grid.y << ")"
              << " | Block: (" << block.x << "," << block.y << ")"
              << std::endl;
}

void vectorAdd(const float *A, const float *B, float *C, int N)
{
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc((void **)&d_A, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice));

    const int block1D = 256; // 1D block size for row/col kernels

    // One2One kernel configuration (2D)
    dim3 blockOne2One(32, 32);
    dim3 gridOne2One((N + blockOne2One.x - 1) / blockOne2One.x,
                     (N + blockOne2One.y - 1) / blockOne2One.y);

    // Row kernel configuration (1D)
    dim3 blockRow(block1D, 1);
    dim3 gridRow((N + blockRow.x - 1) / blockRow.x, 1);

    // Column kernel configuration (1D)
    dim3 blockCol(block1D, 1);
    dim3 gridCol((N + blockCol.x - 1) / blockCol.x, 1);

    benchmark_kernel("One2One", addKernelOne2One, d_A, d_B, d_C, N, gridOne2One, blockOne2One);
    benchmark_kernel("Row", addKernelRow, d_A, d_B, d_C, N, gridRow, blockRow);
    benchmark_kernel("Column", addKernelCol, d_A, d_B, d_C, N, gridCol, blockCol);

    CUDA_CHECK(cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main()
{
    const int N = 1024;
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    std::fill(A, A + N * N, 1.0f);
    std::fill(B, B + N * N, 2.0f);

    vectorAdd(A, B, C, N);

    bool valid = true;
    for (int i = 0; i < N * N; ++i)
    {
        if (fabs(C[i] - 3.0f) > 1e-6)
        {
            valid = false;
            break;
        }
    }
    std::cout << "\nResult validation: " << (valid ? "PASSED" : "FAILED") << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}