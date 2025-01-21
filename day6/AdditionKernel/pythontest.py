import torch
import additionkernel  

input_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).cuda()
additionkernel.addition(input_tensor, input_tensor.size(0))
print("Result after addition:", input_tensor)