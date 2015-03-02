
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
#include "volume_utils.hpp"
extern "C" {
    #include "mkl_vsl.h"
}

namespace zi {
namespace znn {

// 3D convolution using MKL, transpose volume
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

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    // size
    MKL_INT ashape[3]={az,ay,ax}, bshape[3]={bz,by,bx}, rshape[3]={rz,ry,rx};

    // 3d convolution using MKL
    VSLConvTaskPtr task;
    MKL_INT dims=3;
    int status;
    const int mode = VSL_CONV_MODE_DIRECT;//direct convolution
    const int start[3]={bz-1,by-1,bx-1};

    status = vsldConvNewTask(&task,mode,dims,ashape, bshape, rshape);
    status = vslConvSetStart(task, start);
    status = vsldConvExec(task, a.data(), NULL, b.data(), NULL, r.data(), NULL);
    status = vslConvDeleteTask(&task);

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

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    for (int xs=0; xs<stride[0]; xs++)
        for (int ys=0; ys<stride[1]; ys++)
            for (int zs=0; zs<stride[2]; zs++)
            {
                // size
                MKL_INT ashape[3]={az/s[2],ay/s[1],axs/s[0]}, bshape[3]={bz/s[2],by/s[1],bx/s[0]}, rshape[3]={rz/s[2],ry/s[1],rx/s[0]};

                // 3d convolution using MKL
                VSLConvTaskPtr task;
                MKL_INT dims=3;
                int status;
                const int mode = VSL_CONV_MODE_DIRECT;//direct convolution
                const int start[3]={(bz-1)/s[2],(by-1)/s[1],(bx-1)/s[0]};

                // data stride
                int stride[3] = {s[2],s[1],s[0]};
                //MKL_INT input_shape[3]={ax/s[0],ay/s[1],az/s[2]}, kernel_shape[3]={bx/s[0],by/s[1],bz/s[2]};

                // temporal subconvolution output
                double3d_ptr tp = volume_pool.get_double3d(rx,ry,rz);
                double3d& t = *tp;

                // subconvolution
                status = vsldConvNewTask(&task,mode,dims,ashape, bshape, rshape);
                status = vslConvSetStart(task, start);
                status = vsldConvExec(task, a.data()+, stride, b.data(), stride, r.data(), NULL);
                status = vslConvDeleteTask(&task);

                // combine subconvolution results

            }

    return rp;
}


}} // namespace zi::znn
#endif // ZNN_CONV_SSE_HPP_INCLUDED
