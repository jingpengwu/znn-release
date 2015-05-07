#include <boost/python.hpp>
// #include <boost/python/extract.hpp>
// #include <boost/python/numeric.hpp>
// #include <ndarray.hpp>
#include <boost/numpy.hpp>
#include <iostream>

#include "../core/network.hpp"
#include "../front_end/options.hpp"
#include "../../zi/zargs/zargs.hpp"

namespace p = boost::python;
namespace np = boost::numpy;
using namespace zi::znn;

ZiARG_string(options, "", "Option file path");

np::ndarray znn_forward(np::ndarray input_array)
{
    // Initialize the Python runtime.
    Py_Initialize();
    // Initialize NumPy
    np::initialize();

    // options
    // create fake main function parameters
    char* argv[] = { "run znn forward", "--options="forward.config"" };
    int argc=sizeof(argv)/sizeof(argv[0]) - 1;
    zi::parse_arguments(argc, argv);
    options_ptr op = options_ptr(new options(ZiARG_options));
    op->save();
    // create network
    network net(op);

    std::list<double3d_ptr> inputs, outputs;
    inputs[0] = input_array.get_data();

    outputs = net.run_forward(inputs);

    np::dtype dtype = np::dtype::get_builtin<double>();
    std::vector<Py_intptr_t> shape(3);
    int itemsize = dtype.get_itemsize();
    std::vector<Py_intptr_t> strides(3, itemsize);
    np::ndarray output_array = np::from_data( outputs[0].getData(), dtype, p::make_tuple(3), p::make_tuple(sizeof(double)), p::object());
    return output_array;
}

BOOST_PYTHON_MODULE(znn_forward)
{
    def("znn_forward", &znn_forward);
}
