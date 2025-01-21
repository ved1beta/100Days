import torch
import additionkernel  # Ensure the module is built and installed

# Create a tensor on the GPU with int32 type
input_tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32).cuda()

# Call the addition function
additionkernel.addition(input_tensor, input_tensor.size(0))

# Print the result
print("Result after addition:", input_tensor)