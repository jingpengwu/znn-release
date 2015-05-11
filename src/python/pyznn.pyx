import numpy as np
cimport numpy as np

def run_forward(np.ndarray[double, ndim=3, mode="c"] input,\
                np.ndarray[double, ndim=3, mode="c"] output):
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
    znn_forward(&input[0,0,0], iz, iy, ix, &output[0,0,0], oz, oy, ox )
