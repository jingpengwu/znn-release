// #ifndef ZNN_CXX_INCLUDED
// #define ZNN_CXX_INCLUDED

#include "znn.h"

using namespace zi::znn;

inline void prepare_test_c(string ftconf)
{
    // options, create fake main function parameters
    // std::string path = "forward.config";
    options_ptr op = options_ptr(new options( ftconf ));
    op->save();
    // create network
    network net(op);
    net.prepare_testing();
}

inline void pyznn_forward_c(    double* input_py,  unsigned int iz, unsigned int iy, unsigned int ix,
                                double* output_py, unsigned int oz, unsigned int oy, unsigned int ox)
{
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

// #endif // ZNN_CXX_INCLUDED
