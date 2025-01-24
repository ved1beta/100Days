# Project Progress and Tasks

Bro in CUDA 📗 : https://github.com/a-hamdi/cuda

Mentor 🚀 : https://github.com/hkproj
### Mandatory and Optional Tasks
| Day   | Task Description                                                                                     |
|-------|-----------------------------------------------------------------------------------------------------|
| D15   | **Mandatory FA2-Forward**: Implement forward pass for FA2 (e.g., a custom neural network layer).    |
| D20   | **Mandatory FA2-Backwards**: Implement backward pass for FA2 (e.g., gradient computation).          |
| D20   | **Optional Fused Chunked CE Loss + Backwards**: Fused implementation of chunked cross-entropy loss with backward pass. Can use Liger Kernel as a reference implementation. |

---

### Project Progress by Day
| Day   | Files & Summaries                                                                                                                                                                                                                          |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| day1  | **printAdd.cu**: Print global indices for 1D vector (index calculation).<br>**addition.cu**: GPU vector addition; basics of memory allocation/host-device transfer.                                                                 |
| day2  | **function.cu**: Use `__device__` function in kernel; per-thread calculations.                                                                                                                                                       |
| day3  | **addMatrix.cu**: 2D matrix addition; map row/column indices to threads.<br>**anotherMatrix.cu**: Transform matrices with custom function; 2D index operations.                                                                       |
| day4  | **layerNorm.cu**: Layer normalization using shared memory; mean/variance computation.                                                                                                                                                |
| day5  | **vectorSumTricks.cu**: Parallel vector sum via reduction; shared memory optimizations.                                                                                                                                               |
| day6  | **SMBlocks.cu**: Retrieve SM ID per thread via inline PTX.<br>**SoftMax.cu**: Shared-memory softmax; split exponent/normalization steps.<br>**TransposeMatrix.cu**: Matrix transpose via index swapping.<br>**ImportingToPython/rollcall.cu**: Python-CUDA integration.<br>**AdditionKernel/additionKernel.cu**: Modify PyTorch tensors in CUDA. |
| day7  | **naive.cu**: Naive matrix multiplication.<br>**matmul.cu**: Tiled matmul with shared memory.<br>**conv1d.cu**: 1D convolution with shared memory.<br>**pythontest.py**: Validate custom convolution against PyTorch.                               |
| day8  | **pmpbook/chapter3matvecmul.cu**: Matrix-vector multiplication.<br>**pmpbook/chapter3ex.cu**: Benchmarks different matrix add kernels.<br>**pmpbook/deviceinfo.cu**: Prints device properties.<br>**pmpbook/color2gray.cu**: Convert RGB to grayscale.<br>**pmpbook/vecaddition.cu**: Another vector addition example.<br>**pmpbook/imageblur.cu**: Simple image blur.<br>**selfAttention/selfAttention.cu**: Self-attention kernel with online softmax. |
| nvidiadocs | **addition.cu**: 1D/2D vector/matrix addition examples.                                                                                                                                                                                      |
| day9 | flashAttentionFromTut.cu: Minimal Flash Attention kernel with shared memory tiling.<br>bind.cpp: Torch C++ extension bindings for Flash Attention.<br>test.py: Tests the minimal Flash Attention kernel against a manual softmax-based attention for comparison. |