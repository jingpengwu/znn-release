#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <boost/python/numeric.hpp>
#include <boost/numpy.hpp>
#include <iostream>

#include "../core/network.hpp"
#include "../front_end/options.hpp"
#include "../../zi/zargs/zargs.hpp"

namespace p = boost::python;
namespace np = boost::numpy;
using namespace zi::znn;

ZiARG_string(options, "", "Option file path");

np::ndarray znn_forward(np::ndarray input_array, np::ndarray fov, char** argv)
{
    // options, create fake main function parameters
    // char* argv[] = { "run znn forward", "--options=forward.config" };
    int argc=sizeof(argv)/sizeof(argv[0]) - 1;
    zi::parse_arguments(argc, argv);
    options_ptr op = options_ptr(new options(ZiARG_options));
    op->save();
    // create network
    network net(op);

    // the shape of input and output array
    std::size_t iz = input_array.shape(0);
    std::size_t iy = input_array.shape(1);
    std::size_t ix = input_array.shape(2);
    std::size_t oz = iz - fov[0] + 1;
    std::size_t oy = iy - fov[1] + 1;
    std::size_t ox = ix - fov[2] + 1;
    // initialization
    double3d_ptr pinput = volume_pool.get_double3d(ix,iy,iz);
    double3d& input = *pinput;
    // input.data() = input_array.get_data();
    for (std::size_t x=0; x<ix; x++)
        for (std::size_t y=0; y<iy; y++)
            for (std::size_t z=0; z<iz; z++)
                input[x,y,z] = input_array[x,y,z];
    std::list<double3d_ptr> pinputs;
    pinputs.push_back(pinput);
    // prepare output
    std::list<double3d_ptr> poutputs;
    poutputs = net.run_forward(pinputs);
    double3d& output = *(poutputs.front());

    np::dtype dtype = np::dtype::get_builtin<double>();
    p::tuple shape = p::make_tuple(oz,oy,ox);
    // the stride is the bytes of each step in z,y,x direction
    int itemsize = dtype.get_itemsize();
    p::tuple stride= p::make_tuple(oy*ox*itemsize,ox*itemsize,itemsize);
    np::ndarray output_array;
    output_array = np::from_data( output.data(), dtype, shape, stride, p::object());
    return output_array;
}

BOOST_PYTHON_MODULE(znn_forward)
{
    // Initialize the Python runtime.
    Py_Initialize();
    // Initialize NumPy, must be put using Boost.NumPy
    np::initialize();

    def("znn_forward", &znn_forward);
}
