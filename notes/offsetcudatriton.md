 I want to talk about how offsets and stuff is calculated in CUDA and Triton

I will start with CUDA because is easier to explain in my opinion :
```c
__global__ void vectorAdd(const float* A , const float *B, float *C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    if (idx<N){
        C[idx] = A[idx] + B[idx];
    }
}
```
So this is a simple `vectorADD` function who takes as input the following:

`const float* A` -> a constant pointer of type float to the `A` array

`const float* B` -> a constant pointer of type float to the `B` array

`float *C` -> a pointer of type float to the `C` array. Note: `C` pointer is not a constant because we want to modify the element of it

Now lets dive deeper:
`int idx = blockIdx.x * blockDim.x + threadIdx.x` : so we have the position in the grid multiplied by the dimension of each block + position in the thread


Now lets see the triton function:
```python
def __kernelfunction__(input_pointer, output_pointer, N,
                       BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(0) 

    offset = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offset < N

    input_data = tl.load(input_pointer + offset, mask=mask)
    output_data = tl.sqrt(input_data)
    tl.store(output_pointer + offset, output_data, mask=mask)
```
so our `idx` is exactly the `offset` in the triton.
offset will be calculated by the programd_id multiplied by the dimension of the block and we will add an array of [0,1,2,3,....,BLOCKSIZE] .
The result can be thinked that will be an array with each position associated to each thread.
