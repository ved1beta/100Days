#include <torch/extension.h>

void addition(torch::Tensor& input, int arraySize);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("addition", &addition, "Adds 10 to each element of the tensor");
}