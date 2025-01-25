from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

ext_modules = [
    CUDAExtension('example_kernels', 
    [
        'rollcallbinding.cpp',  
        'rollcall.cu', 
    ])
]   

setup(
    name="example_kernels",
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension}
)
