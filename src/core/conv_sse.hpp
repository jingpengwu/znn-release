
//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
//               2015  Jingpeng Wu <jingpeng@princeton.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZNN_CONV_SSE_HPP_INCLUDED
#define ZNN_CONV_SSE_HPP_INCLUDED

#include "types.hpp"
//#include "volume_pool.hpp"
extern "C" {
    #include "mkl_vsl.h"
}

namespace zi {
namespace znn {
/*
bool conv_mkl_1d(double* x, MKL_INT xshape, double* y, MKL_INT yshape, double* r, MKL_INT rshape)
{
    VSLConvTaskPtr task;
    MKL_INT zshape = xshape + yshape - 1;
    double z[zshape];
    int status;

    int mode = VSL_CONV_MODE_AUTO;

    status = vsldConvNewTask1D(&task,mode,xshape,yshape,zshape);
    status = vsldConvExec1D(task,x,1,y,1,z,1);
    status = vslConvDeleteTask(&task);


    // extract the center vector
    /*
    for(int i = 0; i < rshape; i++)
    {
        r[i] = z[i+yshape-1];
    }
    *//*
    r = z+yshape-1;
    return true;

}

// 1D convolution using MKL
inline double3d_ptr bf_conv_mkl_1d(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,1);
    double3d& r = *rp;

    // one dimension vector in y direction for SSE, vector and window
    double *avx=new double[ax], *bvx=new double[bx], *rvx=new double[rx];
    double *avy=new double[ay], *bvy=new double[by], *rvy=new double[ry];
    //double avx[ax], bvx[bx], rvx[rx];
    //double avy[ay], bvy[by], rvy[ry];

    // X
    for ( std::size_t x = 0; x < rx; ++x)
    {
        for ( std::size_t i = 0; i < ay; ++i)
                avy[i] = a[x][i][0];
        for ( std::size_t i = 0; i < by; ++i)
                bvy[i] = b[x][i][0];

        // 1d convolution of av and bv, output: rv
        conv_mkl_1d( avy, ay, bvy, by, rvy, ry );

        for ( std::size_t i = 0; i < ry; ++i )
        {
            r[x][i][0] = rvy[i];
        }
    }

    // Y
    for ( std::size_t y = 0; y < ry; ++y)
    {
        for ( std::size_t i = 0; i < ax; ++i)
                avx[i] = a[i][y][0];
        for ( std::size_t i = 0; i < bx; ++i)
                bvx[i] = b[i][y][0];

        // 1d convolution of av and bv, output: rv
        conv_mkl_1d( avx, ay, bvx, by, rvx, ry );

        for ( std::size_t i = 0; i < ry; ++i )
        {
            r[i][y][0] = rvx[i];
        }
    }

    // combine the X and Y results


    // free the arrays
    delete[] avx, bvx, rvx;
    delete[] avy, bvy, rvy;

    return rp;
}
*/
// this 2D convolution is unfinished
// 2D convolution using MKL
inline double3d_ptr bf_conv_mkl_2d(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;


    // size
    MKL_INT input_shape[2]={ax,ay}, kernel_shape[2]={bx,by};
    MKL_INT tx = ax + bx - 1, ty = ay + by -1;
    MKL_INT tshape[2]={tx,ty};

    // give value
    double input[ax*ay], kernel[bx*by];
    for (std::size_t i = 0; i < ax; ++i)
        for (std::size_t j = 0; j < ay; ++j)
            input[i*ay + j] = a[i][j][0];
    for (std::size_t i = 0; i < bx; ++i)
        for (std::size_t j = 0; j < by; ++j)
            kernel[i*by + j] = b[i][j][0];
    // temporal variable for MKL's long convolution
    double t[tx*ty];

    // 2d convolution using MKL
    VSLConvTaskPtr task;
    MKL_INT rank=2;
    int status;
    int mode = VSL_CONV_MODE_AUTO;

    status = vsldConvNewTask(&task,mode,rank,input_shape, kernel_shape, tshape);
    status = vsldConvExec(task,input,NULL,kernel,NULL,t,NULL);
    status = vslConvDeleteTask(&task);

    // extract the center vector
    double3d_ptr rp = volume_pool.get_double3d(rx,ry,1);
    double3d& r = *rp;
    for(int i = 0; i < rx; i++)
        for(int j = 0; j < ry; j++)
            r[i][j][0] = t[(i+rx-1)*ty+ j+ry-1];

    return rp;
}




}} // namespace zi::znn

#endif // ZNN_CONV_SSE_HPP_INCLUDED
