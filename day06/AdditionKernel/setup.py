from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='additionkernel',
    ext_modules=[
        CUDAExtension(
            name='additionkernel',
            sources=[
                'additionKernelBinding.cpp',
                'additionKernel.cu',
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)