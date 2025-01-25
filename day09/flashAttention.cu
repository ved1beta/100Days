#include <iostream>
#include <cuda_runtime.h>
#include <math_constants.h>
// Q K V = ( N x D) on HBM
// On SRAM -> M
// Bc = M/4d , Br = min(M/4d ,d )
// O = zeros (NxD,)
// l = zeros (N)
// m = - infinity (N)
// we device Q into Tr blocks ,Tr = [N/Br] Q1...Qtr size (Br xD)
// devide K,V into Tc = [N/Bc] of size [Bc x D]
// we device O into Tr block (Br x d)
// l -> device in Tr block Br size
// m -> Tr blocks of Br size
//
// we iterate with j = 1 :Tc:
// Load Kj, Vj from HBM to SRAM
//      iterate over i = 1:Tr:
//      load Qi,Oi,li,mi from HBM on SRAM
//      on chip we compute Sij = Qi*Kj' (Br x Bc)
//      on chip comppute mij = rowmax(Sij),
//      Pij = exp(Sij = mij)
//      Lij = rowsum(Pij) =

#define M 32
#define N 32
#define d 32
#define Bc ((int)ceil(M / (4.0 * d)))
#define Br (min((int)ceil(M / (4.0 * d)), d))
#define Tr ((int)ceil(N / (float)Br))
#define Tc ((int)ceil(N / (float)Bc))

__global__ void forwardKernel(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    float *m,
    float *l)
{
    int threadindex = threadIdx.x;
    
    const int KVBlockSize = Bc * d;
    const int QOBlockSize = Br * d;
    float S[Br * Bc];
    float P[Br * Bc];

    __shared__ float QBlock[QOBlockSize];
    __shared__ float KBlock[KVBlockSize];
    __shared__ float VBlock[KVBlockSize];
    __shared__ float OBlock[QOBlockSize];
    __shared__ float lBlock[Br];
    __shared__ float mBlock[Br];
    
    // float *Qblock = &SRAM[0];
    // float *KBlock = &SRAM[KVBlockSize];
    // float *VBlock = &SRAM[KVBlockSize * 2];
    // float *OBlock = &SRAM[KVBlockSize * 3];

    for (int j = 0; j < Tc; ++j)
    {
        // TC ->
        // load K and V from HBM to SRAM
        for (int p = 0; p < Bc; ++p)
        {
            for (int k = 0; k < d; ++k)
            {
                KBlock[p * d + k] = K[j * KVBlockSize + p * d + k];
                VBlock[p * d + k] = V[j * KVBlockSize + p * d + k];
            }
        }

        for (int i = 0; i < Tr; ++i)
        {
            // Load Qi , Oi, li, mi from HBM to SRAM
            for (int p = 0; p < Br; ++p)
            {   
                for (int k = 0; k < d; ++k)
                {
                    OBlock[p * d + k] = O[j * QOBlockSize + p * d + k];
                    QBlock[p * d + k] = Q[j * QOBlockSize + p * d + k];
                    // now load the l and m
                }
                lBlock[p] = l[threadindex*Br +p]; // Br = 64 . 64th is at 63 [0 1 2 .... 63] [64 65 .. ]
                mBlock[p] = m[threadindex*Br +p];
            }
            //
            // Q1 (Br x D)
            // K1,V1 (Bc x D) => K' (D x Bc)
            //   
            float rowmax = -INFINITY;
            for (int p=0;p<Bc;++p){
                float result=0; // Sij
                for (int k=0;k<d;++k ){ // Br x d  QBlock Brxd Kblock Bcxd
                    result+=QBlock[threadindex*d+ k]*KBlock[k*d+p];  
                }
                S[Br*threadindex+p]=result;
                if(result>rowmax){
                    rowmax = result;
                }
            }
            mBlock[i] = rowmax; // rowmax
            for(int p=0;p<Bc;++p){
                P[Br*threadindex+Bc] = expf(S[Bc*threadindex+p]-rowmax);
                lBlock[i] += P[Bc*threadindex+p];
            }   
            
            // Mblcoks , lblocks
            // m = [ 1. .. .. N ]
            float new_M_max = max(m[i*Br],mBlock[i]);
            float newL = l[i*Br]*expf(m[i*Br]-new_M_max) + lBlock[i]*expf(mBlock[i]-new_M_max);

            // mi -> maximum of rows for N // M
            // mij -> maximum of rows for BR
    }
}

int main()
{

    float Q[N][d], K[N][d], V[N][d], O[N][d];
    float l[N] = {0};
    float m[N] = {-CUDART_INF_F};
    // initialize the matricies
    float *d_Q, *d_K, *d_V, *d_O, *d_N, *d_M, *d_l, *d_m;
    // allocating
    cudaMalloc(&d_Q, N * d * sizeof(float));
    cudaMalloc(&d_K, N * d * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_O, N * d * sizeof(float));
    cudaMalloc(&d_l, N * sizeof(float));
    cudaMalloc(&d_m, N * sizeof(float));
    cudaMalloc(&d_Q, N * d * sizeof(float));
    cudaMalloc(&d_K, N * d * sizeof(float));
    cudaMalloc(&d_V, N * d * sizeof(float));
    cudaMalloc(&d_O, N * d * sizeof(float));
    cudaMalloc(&d_l, N * sizeof(float));
    cudaMalloc(&d_m, N * sizeof(float));
    // sending to the device
    cudaMemcpy(d_Q, Q, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_O, O, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, N * sizeof(float), cudaMemcpyHostToDevice);

#

    // sending results back to host
    cudaMemcpy(O, d_O, N * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);
    return 0;
}