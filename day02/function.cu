#include <iostream>
#include <cuda_runtime.h>

__device__ float square(int x){
    return x*x;
    // __device__ marked function can only be called from anoter device function
    // or a kernel method
}

__global__ void voidKernel(int *input,int *output,int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        output[i] = square(input[i]);
    }   
}


int main(){
    int N = 10; // size of input and output arrays
    int size = N*sizeof(int); // total memory to allocate for the ararys 
    int *h_input = new int[N]; // alocate memory on the CPU
    int *h_output = new int[N]; // alocate memory on the CPU

    for(int i = 0;i<N;i++){
        h_input[i] = i+1;
    }

    int *d_input,*d_output;
    cudaMalloc((void**)&d_input,size);
    cudaMalloc((void**)&d_output,size);

    cudaMemcpy(d_input,h_input,size,cudaMemcpyHostToDevice);
    // destinatin, from,number of bytes to copy, specify that the data is transfered form the host to device

    int threadsPerBlock = 256;
    int blockGrid = (N + threadsPerBlock-1 )/ threadsPerBlock;
    voidKernel<<<blockGrid, threadsPerBlock>>>(d_input, d_output, N);    cudaMemcpy(h_output,d_output,size,cudaMemcpyDeviceToHost);

    std::cout << "Squared array: ";
    for (int i = 0; i < N; i++) {
        std::cout << h_output[i] << " ";
        }
    std::cout << std::endl;

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}