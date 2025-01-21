#include <torch/extension.h>
#include <iostream>

void roll_call_launcher();

void roll_call_binding(){
    roll_call_launcher();
}

PYBIND11_MODULE(example_kernels, m) {
  m.def(
    "rollcall", // Name of the Python function to create
    &roll_call_binding, // Corresponding C++ function to call
    "Launches the roll_call kernel" // Docstring
  );
}
