import torch
import triton
import triton.language as tl

@triton.jit
def __kernelfunction__(input_pointer, output_pointer, N,
                       BLOCKSIZE: tl.constexpr):
    pid = tl.program_id(0)  # Get the program (block) ID

    offset = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offset < N

    input_data = tl.load(input_pointer + offset, mask=mask)
    output_data = tl.sqrt(input_data)
    tl.store(output_pointer + offset, output_data, mask=mask)

def main():
    N = 10

    input_data = torch.arange(0, N, dtype=torch.float32)
    print("Input data:", input_data)

    output_data = torch.empty_like(input_data)

    input_ptr = input_data.to("cuda")
    output_ptr = output_data.to("cuda")

    BLOCKSIZE = 256

    GRID = (triton.cdiv(N, BLOCKSIZE),)

    __kernelfunction__[GRID](input_ptr, output_ptr, N, BLOCKSIZE=BLOCKSIZE)

    output_data = output_ptr.cpu()
    print("Output data:", output_data)

if __name__ == "__main__":
    main()
