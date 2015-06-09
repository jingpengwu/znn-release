import numpy as np
cimport numpy as np

from libcpp.string cimport string


cdef extern from "znn.cpp":
    inline void pyznn_forward_c(    double* input_py,  unsigned int iz, unsigned int iy, unsigned int ix,\
                                    double* output_py, unsigned int oz, unsigned int oy, unsigned int ox)
    inline void feedforward_c( string )
    inline void train_c( string )

def feedforward( ftconf_py ):
    cdef string ftconf_c = ftconf_py
    feedforward_c( ftconf_c )

def train( ftconf_py ):
    cdef string ftconf_c = ftconf_py
    train_c( ftconf_c )

def run_forward(np.ndarray[double, ndim=3, mode="c"] input  not None,\
                np.ndarray[double, ndim=3, mode="c"] output not None):
    """
    run the forward pass of znn
    """
    cdef int iz = input.shape[0]
    cdef int iy = input.shape[1]
    cdef int ix = input.shape[2]
    cdef int oz = output.shape[0]
    cdef int oy = output.shape[1]
    cdef int ox = output.shape[2]
    # cdef string config_fpath = "forward.config"
    pyznn_forward_c(  &input[0,0,0],  iz, iy, ix,
                    &output[0,0,0], oz, oy, ox )
