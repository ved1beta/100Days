#include <torch/extension.h>


void FlashAttention(torch::Tensor &Q,
                    torch::Tensor &K,
                    torch::Tensor &V,
                    torch::Tensor &O,
                    torch::Tensor &m,
                    torch::Tensor &l,
                    const int seq_len,
                    const int head_dim,
                    int Tc, int Tr, int Bc, int Br);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("FlashAttention", &FlashAttention, "FlashAttention forward");
}