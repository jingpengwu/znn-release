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

void znn_forward(   double* input_py,  unsigned int iz, unsigned int iy, unsigned int ix,
                    double* output_py, unsigned int oz, unsigned int oy, unsigned int ox)
{
    // options, create fake main function parameters
    char *argv[] = { "run_znn_forward", "--options=forward.config", NULL };
    int argc=2;
    zi::parse_arguments(argc, argv);
    options_ptr op = options_ptr(new options(ZiARG_options));
    op->save();
    // create network
    network net(op);

    // initialization
    double3d_ptr pinput = volume_pool.get_double3d(ix,iy,iz);
    double3d& input = *pinput;
    // input.data() = input_array.get_data();
    int index = 0;
    for (std::size_t z=0; z<iz; z++)
        for (std::size_t y=0; y<iy; y++)
            for (std::size_t x=0; x<ix; x++)
            {
                input[x][y][z] = input_py[index];
                index++;
            }
    std::list<double3d_ptr> pinputs;
    pinputs.push_back(pinput);

    // prepare output
    std::list<double3d_ptr> poutputs;
    poutputs = net.run_forward(pinputs);
    double3d& output = *(poutputs.front());

    // give value to python numpy output array
    index = 0;
    for (std::size_t z=0; z<oz; z++)
        for (std::size_t y=0; y<oy; y++)
            for (std::size_t x=0; x<ox; x++)
            {
                output_py[index] = output[x][y][z];
                index++;
            }
}
