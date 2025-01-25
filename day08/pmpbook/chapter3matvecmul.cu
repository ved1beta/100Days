#include <iostream>
#include <cuda_runtime.h>

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

__global__ void matrixveckernel(const float *A,const float*b,float*C,const int N){
    // N the size of the NxN A matrix
    // N aslo the size of the vector
    // we need so that each thread will iterate the row 

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // we got 

    if(i<N){
        float result = 0;
        for(int j = 0; j<N;j++){
            result += A[i*N+j] * b[j];
        }
        C[i] = result;
    }
}

void matvecmul(const float*A,const float *b,float*c,const int N){
    float *dA,*db,*dc;
    size_t sizeA = N*N*sizeof(float);
    size_t sizeb = N*sizeof(float);
    int Threads = 256;
    dim3 blockDim(Threads ,1,1);
    dim3 GridDim(ceil(N/(Threads)));


    CUDA_CHECK(cudaMalloc((void**)&dA,sizeA));
    CUDA_CHECK(cudaMemcpy(dA,A,sizeA,cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&db,sizeb));
    CUDA_CHECK(cudaMemcpy(db,b,sizeb,cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&dc,sizeb));

    matrixveckernel<<<GridDim,blockDim>>>(dA,db,dc,N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(c,dc,sizeb,cudaMemcpyDeviceToHost));


    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

}

int main(){
    int N = 1024;
    float *A = new float[N * N];
    float *b = new float[N];

    for(int i = 0 ;i <N;i++){
        b[i] = 1;
        for(int j = 0 ;j<N;j++){
            A[i*N+j] = 1;
        }
    }

    float *c = new float[N];
    matvecmul(A,b,c,N);
    std::cout <<"C[0:10]=[ ";
    for(int i = 0 ;i<10; i++){
        std::cout<<c[i]<<" ";

    }
    std::cout<<"]"<<std::endl;

    free(A);
    free(b);
    free(c);

    return 0;
}


