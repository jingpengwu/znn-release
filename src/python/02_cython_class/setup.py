from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "pyznn",                                # the extesion name
           # the Cython source and additional C++ source files
           sources=["pyznn.pyx", "cstdlib", \
                    "../core/network.hpp", "../front_end/options.hpp", "../core/types.hpp"],
           language="c++",                        # generate and compile C++ code
      )))
