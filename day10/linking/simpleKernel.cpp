#include <torch/extension.h>
#include "ATen/ATen.h"

void cuda_simpleKernel(float *A);

void simpleKernel(at::Tensor A) {
    cuda_simpleKernel(A.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simplekernel", &simpleKernel, "A simple kernel (CUDA)");
}
