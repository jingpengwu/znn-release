from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

import os
os.environ["CC"] = "g++"
os.environ["CXX"] = "g++"

setup(ext_modules = cythonize(Extension(
           "pyznn",                                # the extesion name
           # the Cython source and additional C++ source files
           sources=["pyznn.pyx", "pyznn.cpp"],
           language="c++",                        # generate and compile C++ code
           extra_compile_args=["-std=c++11","-g", "-I../ -I../../src -I../../zi -I/usr/local/boost/1.55.0/boost", "-L../../ -L/usr/local/boost/1.55.0/lib64"],
           extra_link_args=["-lfftw3 -lpthread -lrt -lfftw3_threads", "-std=c++11", "-lboost_program_options -lboost_regex -lboost_filesystem -lboost_system -lboost_timer"]
      )))
