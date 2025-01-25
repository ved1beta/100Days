import triton
import torch
import triton.language as tl


@triton.jit
def addMatrix(Matrix_A,Matrix_B,Matrix_C,sizeX,sizeY,BLOCK_SIZE:tl.constexpr):

    pid_x = tl.program_id(0) # we have the rows
    pid_y = tl.program_id(1) # we have the collumns
    
    row_start = pid_x*BLOCK_SIZE
    col_start = pid_y*BLOCK_SIZE
    
    row_indices = row_start + tl.arange(0,BLOCK_SIZE)
    col_indices = col_start + tl.arange(0,BLOCK_SIZE)
    
    row_indices = row_indices[:,None]
    col_indices = col_indices[None,:]
    
    row_mask = row_indices < sizeY
    col_mask = col_indices < sizeX
    valid_mask = row_mask & col_mask 
    
    flat_indicies = row_indices * sizeX + col_indices
    
    A = tl.load(Matrix_A + flat_indicies,mask =valid_mask,other=0.0)
    B = tl.load(Matrix_B + flat_indicies,mask = valid_mask,other = 0.0)
    
    C = A+B;
    
    tl.store(Matrix_C+flat_indicies,C,mask=valid_mask)


def test_addMatrix():
    sizeX = 8
    sizeY = 8
    BLOCK_SIZE = 2

    Matrix_A = torch.randn(sizeY, sizeX, device='cuda', dtype=torch.float32)
    Matrix_B = torch.randn(sizeY, sizeX, device='cuda', dtype=torch.float32)
    Matrix_C = torch.zeros_like(Matrix_A, device='cuda', dtype=torch.float32)

    Matrix_A_flat = Matrix_A.flatten()
    Matrix_B_flat = Matrix_B.flatten()
    Matrix_C_flat = Matrix_C.flatten()

    grid = (triton.cdiv(sizeX, BLOCK_SIZE), triton.cdiv(sizeY, BLOCK_SIZE))
    addMatrix[grid](Matrix_A_flat, Matrix_B_flat, Matrix_C_flat, sizeX, sizeY, BLOCK_SIZE)

    Matrix_C = Matrix_C_flat.reshape(sizeY, sizeX)

    expected = Matrix_A + Matrix_B
    print("Matrix A:\n", Matrix_A)
    print("Matrix B:\n", Matrix_B)
    print("Matrix C (Triton):\n", Matrix_C)
    print("Expected (PyTorch):\n", expected)
    assert torch.allclose(Matrix_C, expected), "Triton result does not match PyTorch result!"

test_addMatrix()

    
    
    