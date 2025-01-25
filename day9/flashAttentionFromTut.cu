#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
// Grid Dimensions (gridDim.x, gridDim.y):
//     gridDim.x = Number of batches (B).
//     gridDim.y = Number of heads (nh).
// This ensures that each block in the grid handles a specific (batch, head) combination.

// Block Dimensions (blockDim.x):
//     blockDim.x = Number of threads per block.
//     This is typically set to the tile size (e.g., Br for rows or Bc for columns) to divide the workload among threads in the block.

__global__ void flashKernel(const float *Q,
                            const float *K,
                            const float *V,
                            float *O,
                            float *m,
                            float *l,
                            const int seq_len,
                            const int dim_embed,
                            const int Br,
                            const int Bc,
                            const int Tc,
                            const int Tr)
{
    int threadIdxX = threadIdx.x;
    int blockIdxX = blockIdx.x; // batch
    int blockIdxY = blockIdx.y; // head

    int qkvOffset = (blockIdxX * gridDim.y * seq_len * dim_embed) + (blockIdxY * seq_len * dim_embed);
    int lmOffset = (blockIdxX * gridDim.y * seq_len) + (blockIdxY * seq_len);

    extern __shared__ float shared_memory[];
    float *Qi = shared_memory;
    float *Ki = &shared_memory[Br * dim_embed];
    float *Vi = &shared_memory[2 * Br * dim_embed];
    float *Si = &shared_memory[3 * Br * dim_embed];

    for (int r = 0; r < Br; ++r)
    {
        for (int c = 0; c < Bc; ++c)
        {
            int globalRow = blockIdx.x * Br + r;
            int globalCol = blockIdx.y * Bc + c;

            if (globalRow < seq_len && globalCol < seq_len)
            {
                Qi[r * dim_embed + c] = Q[qkvOffset + globalRow * dim_embed + globalCol];
                Ki[r * dim_embed + c] = K[qkvOffset + globalRow * dim_embed + globalCol];
                Vi[r * dim_embed + c] = V[qkvOffset + globalRow * dim_embed + globalCol];
            }
        }
    }

    __syncthreads();

    for (int i = 0; i < Tc; ++i)
    {
        for (int j = 0; j < Tr; ++j)
        {
            int localRowTile = i * Bc + threadIdxX / dim_embed;
            int localColTile = j * Br + threadIdxX % dim_embed;

            if (localRowTile < seq_len && localColTile < seq_len)
            {
                float result = 0.0f;
                for (int k = 0; k < dim_embed; ++k)
                {
                    result += Qi[localRowTile * dim_embed * k] * Ki[dim_embed * k + localColTile];
                }
                result = result * rsqrtf(dim_embed);
                Si[localRowTile * dim_embed + localColTile] = expf(result);
            }
        }
    }

    __syncthreads();
    for (int r = 0; r < Br; ++r)
    {
        for (int c = 0; c < Bc; ++c)
        {
            float scoreSum = 0.0f;
            for (int k = 0; k < dim_embed; ++k)
            {
                scoreSum += Si[r * dim_embed + k] * Vi[k * Bc + c];
            }
            Si[r * dim_embed + c] = scoreSum;
            O[qkvOffset + r * dim_embed + c] = Si[r * dim_embed + c] * Vi[r * dim_embed + c];
        }
    }
}

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = (float)rand() / RAND_MAX;
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


