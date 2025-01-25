import torch
import torch.nn.functional as F
import math

batch_size = 1
n_head = 1
seq_len = 2
head_embd = 2

q = torch.ones(batch_size, n_head, seq_len, head_embd).cuda()
k = torch.ones(batch_size, n_head, seq_len, head_embd).cuda()
v = torch.ones(batch_size, n_head, seq_len, head_embd).cuda()

print('=== profiling manual attention ===')

# Our minimal flash attention aims to be faster than this by avoiding HBM read/writes of N^2 matrices.
def manual_attn(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    manual_result = manual_attn(q, k, v)
print(manual_result)