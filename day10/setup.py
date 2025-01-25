from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='FlashAttention',
    ext_modules=[
        CUDAExtension(
            name='FlashAttention',
            sources=['FlashAttention.cpp', 'FlashAttention.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
