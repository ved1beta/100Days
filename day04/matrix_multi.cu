#include <iostream>
#include <cuda_runtime.h>

__global__ void matmul(const float* A , const float* B , float* C , int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx< N){
        float summ = 0.0f;
        for ( int j ; j<N; j++){
            summ+= A[idx*N + j] * B[j];
        }
        C[idx] = summ;
    }
}

int main(){
    const int N = 10 ;
    float *A , *B, *C;
    
    A = (float *)malloc(N*N* sizeof(float));

    B = (float *)malloc(N*N* sizeof(float));
    C = (float *)malloc(N*N* sizeof(float));

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 1.0f;
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    float *d_a  , *d_b , *d_c ;
    cudaMalloc(&d_a , N*N*sizeof(float));
    cudaMalloc(&d_b , N*N*sizeof(float));
    cudaMalloc(&d_c , N*N*sizeof(float));
    cudaMemcpy(d_a , A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_b , B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    int blocksize=256;
    int gridsize = (N + blocksize - 1) / blocksize;
    matmul<<<gridsize,blocksize>>>(d_a,d_b,d_c,N);
    cudaDeviceSynchronize();
cudaMemcpy(C,d_c,N*sizeof(float),cudaMemcpyDeviceToHost);


}