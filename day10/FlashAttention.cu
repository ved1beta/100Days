#include <cuda_runtime.h>
#include <iostream>
// #include "ATen/ATen.h"
// #include <torch/types.h>

template <typename T>
__global__ void flashKernel(const T *Q,
                            const T *K,
                            const T *V,
                            T *O,
                            T *m,
                            T *l,
                            const int seq_len,
                            const int head_dim,
                            int Tc, int Tr, int Bc, int Br)
{
    // blocks of Bc x Br
    // number of Tc X Tr tiles

    int threadIndex = threadIdx.x;
    int BatchIndex = blockIdx.x;
    int HeadIndex = blockIdx.y;
    int NrHeads = gridDim.y;
    // int NrBatches = gridDim.x;

    int qkvOffset = (BatchIndex * NrHeads * seq_len * head_dim) + (HeadIndex * seq_len * head_dim);
    // l and m are of size [batch_size x num_heads x seq_len]
    int lmOffset = (BatchIndex * NrHeads * seq_len) + (HeadIndex * seq_len);

    extern __shared__ float sharedMemory[];
    int TileSize = Bc * head_dim;
    T *Qi = sharedMemory;
    T *Ki = &sharedMemory[TileSize];
    T *Vi = &sharedMemory[2 * TileSize];
    T *Si = &sharedMemory[3 * TileSize];

    for (int j = 0; j < Tc; ++j)
    {
        // line 6 paper : Load KV in the SRAM
        for (int aux = 0; aux < seq_len; ++aux)
        {
            Ki[threadIndex * seq_len + aux] = K[qkvOffset + j * TileSize + threadIndex * seq_len + aux];
            Vi[threadIndex * seq_len + aux] = V[qkvOffset + j * TileSize + threadIndex * seq_len + aux];
        }
        //
        __syncthreads();

        for (int i = 0; i < Tr; ++i)
        {
            // line 8 paper ; Load Qi, Oi, li, mi, into the SRAM
            for (int aux = 0; aux < seq_len; ++aux)
            {
                Qi[threadIndex * seq_len + aux] = Q[qkvOffset + i * TileSize + threadIndex * seq_len + aux];
            }

            // lest load li and mi;
            float rowPrevMax = m[lmOffset + i * Br + threadIndex];
            float rowPrevSum = l[lmOffset + i * Br + threadIndex];

            // we need to compute now the Sij

            float rowMax = -INFINITY;
            for (int y = 0; i < Bc; ++y)
            {
                float sum = 0;
                for (int x = 0; x < head_dim; ++x)
                {
                    sum += Qi[y * seq_len + x] * Ki[x * seq_len + y];
                }
                sum = sum * rsqrtf(head_dim);
                Si[Br * threadIndex + y] = sum;
                rowMax = fmaxf(rowMax, sum);
            }

            // we calcultaed the row max + Sij
            float rowSum = 0;
            for (int aux = 0; aux < Bc; ++aux)
            {
                Si[threadIndex * Br + aux] = expf(Si[threadIndex * Br + aux] - rowMax);
                rowSum += Si[threadIndex * Br + aux];
            }

            float newMax = fmaxf(rowPrevMax, rowMax);
            float newSum = rowPrevSum * expf(rowPrevMax - newMax) + rowSum * expf(rowMax - newMax);

            // we need to iterate over all the elements
            for (int aux = 0; aux < head_dim; ++aux)
            {
                float value = 0.0f;
                for (int y = 0; y < Bc; ++y)
                {
                    value += Si[threadIndex * Bc + y] * Vi[y * head_dim + aux];
                }
                O[qkvOffset + (TileSize * i) + (threadIndex * seq_len) + aux] = (1 / newSum) * ((rowPrevSum * expf(rowPrevMax - newMax) * value) * O[qkvOffset + (TileSize * i) + (threadIndex * head_dim + aux)] + (expf(rowMax - newMax) * value));
            }

            l[lmOffset + i * Br + threadIndex] = newSum;
            m[lmOffset + i * Br + threadIndex] = newMax;
        }
        __syncthreads();
    }
}

// void FlashAttention(torch::Tensor &Q,
//                     torch::Tensor &K,
//                     torch::Tensor &V,
//                     torch::Tensor &O,
//                     torch::Tensor &m,
//                     torch::Tensor &l,
//                     const int seq_len,
//                     const int head_dim,
//                     int Tc, int Tr, int Bc, int Br)
// {
//     const int batch_size = Q.size(0);
//     const int num_heads = Q.size(1);
//     const int seq_len = Q.size(2);
//     const int head_dim = Q.size(3);
//     const int Bc = 32;
//     const int Br = 32;
//     const int Tc = ceil((float)seq_len / Bc);
//     const int Tr = ceil((float)seq_len / Br);
//     int sharedMemorySize = 3 * Bc * Br * head_dim * sizeof(float) + Bc * Br * sizeof(float);
//     dim3 grid(batch_size, num_heads);
//     dim3 block(Bc);

//     AT_DISPATCH_FLOATING_TYPES(Q.type(), "flashKernel", [&]
//                                { flashKernel<<<grid, block, sharedMemorySize>>>(Q.data_ptr<scalar_t>(),
//                                                                                 K.data_ptr<scalar_t>(),
//                                                                                 V.data_ptr<scalar_t>(),
//                                                                                 O.data_ptr<scalar_t>(),
//                                                                                 m.data_ptr<scalar_t>(),
//                                                                                 l.data_ptr<scalar_t>(),
//                                                                                 seq_len, head_dim, Tc, Tr, Bc, Br); });
//     cudaDeviceSynchronize();
//     auto err = cudaGetLastError();
//     if (err != cudaSuccess)
//     {
//         TORCH_CHECK(false, "CUDA error: ", cudaGetErrorString(err));
//     }
// }


// torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V)
// {
//     const int batch_size = Q.size(0);
//     const int num_heads = Q.size(1);
//     const int seq_len = Q.size(2);
//     const int head_dim = Q.size(3);

//     dim3 grid(batch_size, num_heads);
//     dim3 block(Bc);

//     auto O = torch::zeros_like(Q);
//     auto l = torch::zeros({batch_size, num_heads, seq_len});
//     auto m = torch::zeros({batch_size, num_heads, seq_len});

//     torch::Device device(torch::kCUDA);
//     l = l.to(device);
//     m = m.to(device);

//     auto m = torch::full({batch_size, num_heads, seq_len}, -INFINITY);
//     int sharedMemorySize = 3 * Bc * Br * head_dim * sizeof(float) + Bc * Br * sizeof(float);

//     flashKernel<<<grid, block, sharedMemorySize>>>(Q.data_ptr<float>(),
//                                                    K.data_ptr<float>(),
//                                                    V.data_ptr<float>(),
//                                                    O.data_ptr<float>(),
//                                                    m.data_ptr<float>(),
//                                                    l.data_ptr<float>(),
//                                                    seq_len, head_dim, Tc, Tr, Bc, Br);

//     return O;
// }


void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = 1;
    }
}

int main()
{
    const int batch_size = 1;
    const int seq_len = 2;
    const int nr_heads = 1;
    const int dim_head = 2;
    const int Bc = 32;
    const int Br = 32;
    const int Tc = ceil((float)seq_len / Bc);
    const int Tr = ceil((float)seq_len / Br);
    float *Q = new float[batch_size * nr_heads * seq_len * dim_head];
    float *K = new float[batch_size * nr_heads * seq_len * dim_head];
    float *V = new float[batch_size * nr_heads * seq_len * dim_head];
    float *O = new float[batch_size * nr_heads * seq_len * dim_head];
    float *m = new float[batch_size * nr_heads * seq_len];
    float *l = new float[batch_size * nr_heads * seq_len];
    const int sharedMemory = (3 * Bc * dim_head * sizeof(float)) + (Bc * Br * sizeof(float));

    randomInit(Q, batch_size * nr_heads * seq_len * dim_head);
    randomInit(K, batch_size * nr_heads * seq_len * dim_head);
    randomInit(V, batch_size * nr_heads * seq_len * dim_head);


    float *dQ;
    float *dK;
    float *dV;
    float *dO;
    float *dm;
    float *dl;

    cudaMalloc(&dQ, batch_size * nr_heads * seq_len * dim_head * sizeof(float));
    cudaMalloc(&dK, batch_size * nr_heads * seq_len * dim_head * sizeof(float));
    cudaMalloc(&dV, batch_size * nr_heads * seq_len * dim_head * sizeof(float));
    cudaMalloc(&dO, batch_size * nr_heads * seq_len * dim_head * sizeof(float));
    cudaMalloc(&dm, batch_size * nr_heads * seq_len * sizeof(float));
    cudaMalloc(&dl, batch_size * nr_heads * seq_len * sizeof(float));
    
    cudaMemcpy(dQ, Q, batch_size * nr_heads * seq_len * dim_head * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, K, batch_size * nr_heads * seq_len * dim_head * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V, batch_size * nr_heads * seq_len * dim_head * sizeof(float), cudaMemcpyHostToDevice);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);


    dim3 gridSize(batch_size,nr_heads);
    dim3 blockSize(Bc);

    flashKernel<<<gridSize, blockSize, sharedMemory>>>(Q,
                                                       K,
                                                       V,
                                                       O,
                                                       m,
                                                       l,
                                                       seq_len,
                                                       dim_head,
                                                       Br,
                                                       Bc,
                                                       Tc,
                                                       Tr);

    cudaMemcpy(O, dO, batch_size * nr_heads * seq_len * dim_head * sizeof(float), cudaMemcpyDeviceToHost);


    int firstBatch = 1;
    int firstHead = 1;
    
    printf("Printing the first element of the output tensor\n");
    for(int i = 0; i < seq_len; i++)
    {
        printf("O[%d][%d][%d][%d] = %f\n", firstBatch, firstHead, i, 0, O[firstBatch * nr_heads * seq_len * dim_head + firstHead * seq_len * dim_head + i * dim_head]);
    }

    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
    cudaFree(dO);
    cudaFree(dm);
    cudaFree(dl);
    free(Q);
    free(K);
    free(V);
    free(O);
    free(m);
    free(l);

    return 0;
}


