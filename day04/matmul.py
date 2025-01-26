import torch
import time

# Set matrix size
N = 1024

# Create matrices on GPU
A = torch.ones(N, N, device='cuda')
B = torch.full((N, N), 2.0, device='cuda')

# Warm-up
for _ in range(10):
    torch.mm(A, B)

# Timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
C = torch.mm(A, B)
end.record()
torch.cuda.synchronize()

print(f"PyTorch Execution Time: {start.elapsed_time(end):.3f} ms")

# Verify result
expected = 2 * N * torch.ones(N, N, device='cuda')
print(f"Correctness: {torch.allclose(C, expected, atol=0.1)}")