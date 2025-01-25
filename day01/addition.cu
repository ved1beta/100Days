#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A , const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    // so blockIdx.x -> is the ID of thread
    // block dim = the size of the window we work on it
    // threaidx =
    if (idx<N){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){
    const int N = 1024; // elements in vec
    const int size = N * sizeof(int); // total size of vectors in bytes

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for(int i = 0 ;i <N;i++){
        h_A[i] = 1;
        h_B[i] = i;
    }

    float *d_A, *d_B,*d_C;

    cudaMalloc((void**)&d_A,size);
    cudaMalloc((void**)&d_B,size);
    cudaMalloc((void**)&d_C,size);

    // copy input data from host to device:
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = ( N + threadsPerBlock -1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    for(int i =N-10;i<N;i++){
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A; 
    delete[] h_B; 
    delete[] h_C; 

    return 0; 
}
