#include <cuda_runtime.h>
#include <iostream>

// Using smaller dimensions for visualization
#define WIDTH 4
#define HEIGHT 4

__global__ void transposeMatrix(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int inputIndex = y * width + x;
        int outputIndex = x * height + y;
        output[outputIndex] = input[inputIndex];
    }
}

void printMatrix(const float* matrix, int width, int height, const char* label) {
    std::cout << "\n" << label << ":\n";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%4.0f ", matrix[i * width + j]);
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void checkCudaError(const char* message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << message << " - CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialize input matrix with sequential numbers
    for (int i = 0; i < width * height; i++) {
        h_input[i] = static_cast<float>(i);
    }

    // Print initial matrix
    printMatrix(h_input, width, height, "Input Matrix");

    // Allocate device memory
    float* d_input;
    float* d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    std::cout << "Data copied to GPU\n";

    // Define block and grid sizes
    dim3 blockSize(2, 2);  // Smaller block size for small matrix
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    std::cout << "Launching kernel with grid size: (" << gridSize.x << "," 
              << gridSize.y << "), block size: (" << blockSize.x << "," 
              << blockSize.y << ")\n";

    // Launch kernel
    transposeMatrix<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    checkCudaError("Kernel execution failed");

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print transposed matrix
    printMatrix(h_output, height, width, "Transposed Matrix");

    // Print the transformation for each element
    std::cout << "Element Movement During Transposition:\n";
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int inputIndex = i * width + j;
            int outputIndex = j * height + i;
            printf("Element at (%d,%d)[%d] moves to (%d,%d)[%d]\n", 
                   i, j, inputIndex, j, i, outputIndex);
        }
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}