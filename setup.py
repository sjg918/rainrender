
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rainrenderongpu',
    ext_modules=[
        CUDAExtension('rr_cuda', [
            'rainrenderongpu/render_cpp.cpp',
            'rainrenderongpu/render_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})

# & C:/Users/sgj51/anaconda3/python.exe d:/raindrop/setup.py build develop