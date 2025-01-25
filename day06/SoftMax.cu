#include <iostream>
#include <cuda_runtime.h>

__global__ void SoftMaxNaive(float *input,float *output,int size){
    int numThreads = blockDim.x;
    

    //each thread to compute softmax for this:
    int numElementsPerThread = size/numThreads;

    int threadIndex = threadIdx.x;

    int startIndex = threadIndex * numElementsPerThread;
    int endIndex = min(size,startIndex* numElementsPerThread);


    float MaxValue = 0.0;
    for (int i = 0; i < size; i++) {
        if (input[i] > MaxValue) {
            MaxValue = input[i];
        }
    }

    float sumExp = 0.0;
    for(int i =0;i<size;i++){
        sumExp +=expf(input[i] - MaxValue);
    }

    for(int i =startIndex;i<endIndex;i++){
        output[i] = expf(input[i] - MaxValue) / sumExp;
    }
}

__global__ void SoftMaxShared(float *input,float *output,int size){
    int numThreads = blockDim.x;
    int numElementsPerThread = size/numThreads;
    int threadIndex = threadIdx.x;
    int startIndex = threadIndex * numElementsPerThread;
    int endIndex = min(size,startIndex* numElementsPerThread);


    /// Calculate the Maximum 
    __shared__ float SharedMaxValue[numThreads];
    float MaxValue = 0.0;
    for (int i = startIndex; i < endIndex; i++) {
        if (input[i] > MaxValue) {
            MaxValue = input[i];
        }
    }
    SharedMaxValue[threadIndex] = MaxValue;
    __syncthreads();
    for (int i = 0; i < numThreads; i++) {
        if (SharedMaxValue[i] > MaxValue) {
            MaxValue = SharedMaxValue[i];
        }
    }
    

    /// Now we need to calcualte the SumExp
    __shared__ float sharedSumExp[numThreads];
    float sumExp = 0.0;
    for(int i =startIndex;i<endIndex;i++){
        sumExp +=expf(input[i] - MaxValue);
    }
    sharedSumExp[threadIndex] = sumExp;
    __syncthreads();

    for(int i = 0;i<numThreads;i++){
        sumExp+= sharedSumExp[i];
    }

    for (int i = startIndex; i < endIndex; i++) {
        output[i] = expf(input[i] - MaxValue) / sumExp;
    }
}