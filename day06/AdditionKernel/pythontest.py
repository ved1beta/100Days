import torch
import additionkernel  

input_tensor = torch.randn(100).cuda()
additionkernel.addition(input_tensor, input_tensor.size(0))
print("Result after addition:", input_tensor)