
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

    // 3d convolution using MKL
    VSLConvTaskPtr task;
    MKL_INT dims=3;
    int status;
    const int mode = VSL_CONV_MODE_DIRECT;//direct convolution--DIRECT, FFT

    // the kernel
    MKL_INT tbshape[3]={(bz-1)/s[2]+1, (by-1)/s[1]+1, (bx-1)/s[0]+1};
    //std::cout<<"tbshape: " << tbshape[0]<<", "<<tbshape[1]<<", "<<tbshape[2]<<std::endl;
    double tb[ tbshape[0]* tbshape[1]* tbshape[2] ];
    for (int x=bx-1, xt=tbshape[2]-1; x>=0; x-=s[0], xt--)
        for (int y=by-1, yt=tbshape[1]-1; y>=0; y-=s[1], yt--)
            for(int z=bz-1, zt=tbshape[0]-1; z>=0; z-=s[2], zt--)
                tb[zt + yt*tbshape[0] + xt*tbshape[1]*tbshape[0]] = b[x][y][z];
    int start[3]={tbshape[0]-1,tbshape[1]-1,tbshape[2]-1};
    //std::cout<< "start: "<<start[0]<<", "<<start[1]<<", "<<start[2]<<std::endl;

    // temporal volume size
    MKL_INT tashape[3]={(az-1)/s[2]+1, (ay-1)/s[1]+1, (ax-1)/s[0]+1};
    MKL_INT trshape[3]={tashape[0]-tbshape[0]+1, tashape[1]-tbshape[1]+1, tashape[2]-tbshape[2]+1};

    // temporal subconvolution output
    double ta[ tashape[0]* tashape[1]* tashape[2] ];
    double tr[ trshape[0]* trshape[1]* trshape[2] ];

    // sparseness
    for (int xs=0; xs<s[0]; xs++)
        for (int ys=0; ys<s[1]; ys++)
            for (int zs=0; zs<s[2]; zs++)
            {
                // temporal volume size
                MKL_INT tashape[3]={(az-zs-1)/s[2]+1, (ay-ys-1)/s[1]+1, (ax-xs-1)/s[0]+1};
                MKL_INT trshape[3]={tashape[0]-tbshape[0]+1, tashape[1]-tbshape[1]+1, tashape[2]-tbshape[2]+1};

                // prepare input
                for (std::size_t x=xs, xt=0; x<ax; x+=s[0], xt++)
                    for (std::size_t y=ys, yt=0; y<ay; y+=s[1], yt++)
                        for(std::size_t z=zs, zt=0; z<az; z+=s[2], zt++)
                            ta[ zt+ yt*tashape[0] + xt*tashape[1]*tashape[0] ] = a[x][y][z];

                // subconvolution
                //std::cout<<"subconvolution..."<<std::endl;
                status = vsldConvNewTask(&task,mode,dims,tashape, tbshape, trshape);
                //std::cout<<"status-->new task:          "<<status<<std::endl;
                status = vslConvSetStart(task, start);
                //std::cout<<"status-->set start:         "<<status<<std::endl;
                status = vsldConvExec(task, ta, NULL, tb, NULL, tr, NULL);
                //std::cout<<"status-->conv exec:         "<<status<<std::endl;
                status = vslConvDeleteTask(&task);
                //std::cout<<"status-->conv delete task:  "<<status<<std::endl;

                // combine subconvolution results
                for (std::size_t x=xs, wx=0; x<rx; x+=s[0], wx++)
                    for (std::size_t y=ys, wy=0; y<ry; y+=s[1], wy++ )
                        for (std::size_t z=zs, wz=0; z<rz; z+=s[2], wz++)
                            r[x][y][z] = tr[wz + wy*trshape[0] + wx*trshape[1]*trshape[0] ];
            }

    return rp;
}

}} // namespace zi::znn
#endif // ZNN_CONV_SSE_HPP_INCLUDED
