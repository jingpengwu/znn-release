
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

// 3D convolution using MKL
inline double3d_ptr bf_conv_mkl(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;
    std::size_t rz = az - bz + 1;

    // size
    MKL_INT input_shape[3]={ax,ay,az}, kernel_shape[3]={bx,by,bz};
    MKL_INT tx = ax + bx - 1, ty = ay + by - 1, tz = az + bz - 1;
    MKL_INT tshape[3]={tx,ty,tz};

    // give value
    double* input=a.data();
    double* kernel=b.data();

    // temporal variable for MKL's long convolution
    double t[tx*ty*tz];

    // 2d convolution using MKL
    VSLConvTaskPtr task;
    MKL_INT dims=3;
    int status;
    const int mode = VSL_CONV_MODE_DIRECT;//direct convolution
    //const int start[3]={bx-1,by-1,bz-1};

    status = vsldConvNewTask(&task,mode,dims,input_shape, kernel_shape, tshape);
    //status = vslConvSetStart(task, start);
    status = vsldConvExec(task,input,NULL,kernel,NULL,t,NULL);
    status = vslConvDeleteTask(&task);

    // extract the center vector
    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;
    for(int i = 0; i < rx; i++)
        for(int j = 0; j < ry; j++)
            for (int k = 0; k < rz; ++k)
                r[i][j][k] = t[(i+bx-1)*ty*tz+ (j+by-1)*tz + k+bz-1];

    return rp;
}

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
    double* input=a.data();
    double* kernel=b.data();
    // temporal variable for MKL's long convolution
    double t[tx*ty];

    // 2d convolution using MKL
    VSLConvTaskPtr task;
    MKL_INT dims=2;
    int status;
    int mode = VSL_CONV_MODE_DIRECT;

    status = vsldConvNewTask(&task,mode,dims,input_shape, kernel_shape, tshape);
    status = vsldConvExec(task,input,NULL,kernel,NULL,t,NULL);
    status = vslConvDeleteTask(&task);

    // extract the center vector
    double3d_ptr rp = volume_pool.get_double3d(rx,ry,1);
    double3d& r = *rp;
    for(int i = 0; i < rx; i++)
        for(int j = 0; j < ry; j++)
            r[i][j][0] = t[(i+bx-1)*ty+ j+by-1];

    return rp;
}

inline double3d_ptr bf_conv_sparse_mkl( const double3d_ptr& ap,
                                        const double3d_ptr& bp,
                                        const vec3i& s)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax - bx + 1;
    std::size_t ry = ay - by + 1;
    std::size_t rz = az - bz + 1;

    // size
    MKL_INT input_shape[3]={ax,ay,az}, kernel_shape[3]={bx,by,bz};
    MKL_INT tx = ax + bx - 1, ty = ay + by - 1, tz = az + bz - 1;
    MKL_INT tshape[3]={tx,ty,tz};

    // give value
    double* input=a.data();
    double* kernel=b.data();
    // temporal variable for MKL's long convolution

    // get stride
    int stride[3] = {s[0],s[1],s[2]};

    double t[tx*ty*tz];

    // 2d convolution using MKL
    VSLConvTaskPtr task;
    MKL_INT dims=3;
    int status;
    const int mode = VSL_CONV_MODE_DIRECT;//direct convolution
    //const int start[3]={bx-1,by-1,bz-1};

    status = vsldConvNewTask(&task,mode,dims,input_shape, kernel_shape, tshape);
    //status = vslConvSetStart(task, start);
    status = vsldConvExec(task, input, NULL, kernel, NULL, t, NULL);
    status = vslConvDeleteTask(&task);

    // extract the center vector
    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;
    for(int i = 0; i < rx; ++i)
        for(int j = 0; j < ry; ++j)
            for (int k = 0; k < rz; ++k)
                r[i][j][k] = t[(i*s[0]+bx-1)*ty*tz+ (j*s[1]+by-1)*tz + k*s[2]+bz-1];

    return rp;
}

}} // namespace zi::znn

#endif // ZNN_CONV_SSE_HPP_INCLUDED
