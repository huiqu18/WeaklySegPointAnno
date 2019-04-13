from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='crfloss_cpp',
    ext_modules=[
        CppExtension('crfloss_cpp', ['DenseCRFLoss.cpp', 'permutohedral.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
