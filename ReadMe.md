# Project Summary

## day1
**printAdd.cu**  
- Summary: Prints the global index (block + thread) for a 1D vector.  
- Learned: How to calculate global indices in CUDA using blockIdx.x and threadIdx.x.

**addition.cu**  
- Summary: Adds elements from two vectors on the GPU.  
- Learned: Basics of GPU memory allocation and memory transfers between host and device.

## day2
**function.cu**  
- Summary: Demonstrates a __device__-defined function (square) called within a kernel.  
- Learned: Separating device functions from kernels and using them for per-thread calculations.

## day3
**addMatrix.cu**  
- Summary: Adds two matrices using a 2D grid and block arrangement.  
- Learned: Mapping row and column indices to each thread for parallel operations.

**anotherMatrix.cu**  
- Summary: Uses a custom device function (randomFunction) to transform elements in two matrices.  
- Learned: Creating custom operations per thread, referencing 2D indices.

## day4
**layerNorm.cu**  
- Summary: Implements layer normalization using shared memory for intermediate sums.  
- Learned: Using shared memory for partial sums, computing mean/variance, and normalizing data.

## day5
**vectorSumTricks.cu**  
- Summary: Performs a reduction to sum vector elements in parallel with shared memory.  
- Learned: Parallel reductions, chunking large arrays, and partial sums in shared memory.

## day6
**SMBlocks.cu**  
- Summary: Retrieves the streaming multiprocessor (SM) ID for each thread.  
- Learned: Using inline PTX to identify which SM executes each thread.

**SoftMax.cu**  
- Summary: Demonstrates naive vs. shared-memory-based softmax on the GPU.  
- Learned: Splitting exponent and normalization steps across threads, using shared memory effectively.

**TransposeMatrix.cu**  
- Summary: Transposes a matrix by swapping row and column indices in a 2D thread grid.  
- Learned: Converting row-major indices for transposition and mapping threads to coordinate pairs.

**ImportingToPython/rollcall.cu**  
- Summary: Launches a kernel that prints a simple message and numeric value from each thread.  
- Learned: Simple Python-CUDA integration, synchronization, and debugging prints on the GPU.

**AdditionKernel/additionKernel.cu**  
- Summary: Adds a constant offset to each element in a PyTorch tensor via CUDA.  
- Learned: Accessing PyTorch tensor data pointers from CUDA for in-place operations.

## nvidiadocs
**addition.cu**  
- Summary: Various examples illustrating vector and matrix addition kernels with different launch configurations.  
- Learned: Progression from basic 1D vector addition to more complex 2D matrix addition patterns.