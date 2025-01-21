#include <iostream>
#include <cuda_runtime.h>

__global__ void roll_call_kernel() {
	const int threadIndex = threadIdx.x;
	printf("Thread %d here!\n", threadIndex);
    printf("Te iubesc atat de mult: %d \n",threadIndex*1000);
}

void roll_call_launcher() {
    roll_call_kernel<<<1, 5>>>();
    cudaDeviceSynchronize();
}

int main() {
    roll_call_launcher();
	return 0;
}