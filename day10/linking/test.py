from torch.utils.cpp_extension import load

simplekernel = load(
    name='simplekernel',
    sources=['simpleKernel.cpp', 'simpleKernel.cu'],
    verbose=True
)

# Test kernel
import torch
A = torch.zeros(32, device='cuda', dtype=torch.float32)
simplekernel.simplekernel(A)
print(A)
