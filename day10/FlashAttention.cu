#include <cuda_runtime.h>
#include <iostream>

__global__ void flashKernel(const float *Q,
                            const float *K,
                            const float *V,
                            float *O,
                            float *m,
                            float *l,
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
    int NrBatches = gridDim.x;

    int qkvOffset = (BatchIndex * NrHeads * seq_len * head_dim) + (HeadIndex * seq_len * head_dim);
    // l and m are of size [batch_size x num_heads x seq_len]
    int lmOffset = (BatchIndex * NrHeads * seq_len) + (HeadIndex * seq_len);

    extern __shared__ float sharedMemory[];
    int TileSize = Bc * head_dim;
    float *Qi = sharedMemory;
    float *Ki = &sharedMemory[TileSize];
    float *Vi = &sharedMemory[2 * TileSize];
    float *Si = &sharedMemory[3 * TileSize];

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
                O[qkvOffset + (TileSize * i) + (threadIndex * seq_len) + aux] = (1 / newSum) *
                                                                                ((rowPrevSum * expf(rowPrevMax - newMax) * value) * O[qkvOffset + (TileSize * i) \ 
                                                                                + (threadIndex * head_dim + aux)] + (expf(rowMax - newMax) * value));
            }

            l[lmOffset + i * Br + threadIndex] = newSum;
            m[lmOffset + i * Br + threadIndex] = newMax;
        }
        __syncthreads();
    }
}