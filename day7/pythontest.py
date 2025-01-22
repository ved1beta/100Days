import torch
import numpy as np

def test_conv1d():
    input_size = 1024
    kernel_size = 5
    kernel_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, -2.0], dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_cpu = torch.ones(1, 1, input_size, dtype=torch.float32)  #
    x_gpu = x_cpu.to(device)

    torch_conv = torch.nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        padding=kernel_size // 2, 
        bias=False,
        padding_mode="zeros",
    ).to(device)

    with torch.no_grad():
        torch_conv.weight.data = kernel_weights.view(1, 1, kernel_size).to(device)

    y_torch = torch_conv(x_gpu)
    y_torch_np = y_torch.detach().cpu().squeeze().numpy()

    print("PyTorch Output (first 10):", y_torch_np[:10])

if __name__ == "__main__":
    test_conv1d()