#include <cuda_runtime.h>
#include <iostream>

int main()
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    std::cout << "Devices are : " << dev_count << std::endl;

    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count; ++i)
    {
        cudaGetDeviceProperties(&dev_prop, i);
    }
    std::cout << "Max Threads per Block : " << dev_prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per MultiProcessor :" << dev_prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Blocks per MultiProcessor : " << dev_prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "Clock rate : " << dev_prop.clockRate << std::endl;
    std::cout << "Max Grid Size (X,Y,Z) : (" << dev_prop.maxGridSize[0] << "," << dev_prop.maxGridSize[1] << "," << dev_prop.maxGridSize[2] << ")" << std::endl;
    std::cout << "Max Threads Dim (X,Y,Z) : (" << dev_prop.maxThreadsDim[0] << "," << dev_prop.maxThreadsDim[1] << "," << dev_prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Shared Memory per Block : " << dev_prop.sharedMemPerBlock << std::endl;
    std::cout << "Max Shared Memory per MultiProcessor : " << dev_prop.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Max Registers per Block : " << dev_prop.regsPerBlock << std::endl;
    std::cout << "Max Registers per MultiProcessor : " << dev_prop.regsPerMultiprocessor << std::endl;
    std::cout << "Warp Size : " << dev_prop.warpSize << std::endl;
    std::cout << "Max Threads per Warp : " << dev_prop.maxThreadsPerMultiProcessor / dev_prop.warpSize << std::endl;
    std::cout << "Max Warps per MultiProcessor : " << dev_prop.maxThreadsPerMultiProcessor / dev_prop.warpSize << std::endl;
    std::cout << "Max Warps per Block : " << dev_prop.maxThreadsPerBlock / dev_prop.warpSize << std::endl;
    std::cout << "Max Warps per Grid : " << dev_prop.maxThreadsPerBlock / dev_prop.warpSize * dev_prop.maxGridSize[0] * dev_prop.maxGridSize[1] * dev_prop.maxGridSize[2] << std::endl;
    std::cout << "Max Warps per Device : " << dev_prop.maxThreadsPerBlock / dev_prop.warpSize * dev_prop.maxGridSize[0] * dev_prop.maxGridSize[1] * dev_prop.maxGridSize[2] * dev_prop.multiProcessorCount << std::endl;
    std::cout << "Max Blocks per Device : " << dev_prop.maxBlocksPerMultiProcessor * dev_prop.multiProcessorCount << std::endl;
    std::cout << "Max Threads per Device : " << dev_prop.maxThreadsPerBlock * dev_prop.multiProcessorCount << std::endl;
    std::cout << "Max Warps per Device : " << dev_prop.maxThreadsPerBlock / dev_prop.warpSize * dev_prop.maxGridSize[0] * dev_prop.maxGridSize[1] * dev_prop.maxGridSize[2] * dev_prop.multiProcessorCount << std::endl;

}