#include <cuda_runtime.h>
#include <iostream>

__global__ void imageblurkernel(const float *A, float *C, const int sizeArray, const int sizeKernel)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = sizeKernel / 2;

    //   1  2  3  2
    //   4  5  6  2
    //   1  2  3  2
    //   5  6  7  2
    //
    //   Sow we lets say we are at index = 1 first element
    //   we need now to do this :
    //   we only use the blur when if it dosnt overflow
    if (i < sizeArray && j < sizeArray)
    {
        float PixelValue = 0.0;
        int pixels = 0;
        for (int blurRow = -radius; i <= radius; i++)
        {
            for (int blurCol = -radius; j <= radius; j++)
            {
                // so now we are in the kernel
                int curRow = i + blurRow;
                int curCol = j + blurCol;

                if (curRow < 0 || curRow >= sizeArray || curCol < 0 || curCol >= sizeArray)
                {
                    PixelValue += A[curRow * sizeArray + curCol];
                    pixels++;
                }
            }
        }
        C[sizeArray * j + i] = PixelValue / pixels;
    }
}