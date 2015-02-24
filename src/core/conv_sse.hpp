
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

bool conv_mkl_1d(double* x, MKL_INT xshape, double* y, MKL_INT yshape, double* r, MKL_INT rshape)
{
    VSLConvTaskPtr task;
    MKL_INT zshape = xshape + yshape - 1;
    double z[zshape];
    int status;

    int mode = VSL_CONV_MODE_AUTO;

    /*
    *  Create task descriptor (create descriptor of problem)
    */
    status = vsldConvNewTask1D(&task,mode,xshape,yshape,zshape);
    if( status != VSL_STATUS_OK ){
       printf("ERROR: creation of job failed, exit with %d\n", status);
       return 1;
    }

    /*
    *  Execute task (Calculate 1 dimension convolution of two arrays)
    */
    status = vsldConvExec1D(task,x,1,y,1,z,1);
    if( status != VSL_STATUS_OK ){
       printf("ERROR: job status bad, exit with %d\n", status);
       return 1;
    }

    /*
    *  Delete task object (delete descriptor of problem)
    */
    status = vslConvDeleteTask(&task);
    if( status != VSL_STATUS_OK ){
       printf("ERROR: failed to delete task object, exit with %d\n", status);
       return 1;
    }

    // extract the center vector
    /*
    for(int i = 0; i < rshape; i++)
    {
        r[i] = z[i+yshape-1];
    }
    */
    r = z+yshape-1;
    return true;

}

bool conv_mkl_2d(double* x, MKL_INT x1, MKL_INT x0, double* y, MKL_INT y1, MKL_INT y0, double* r, MKL_INT r1, MKL_INT r0)
//bool conv_sse_2d(double* x, MKL_INT* xshape, double* y, MKL_INT* yshape, double* r, MKL_INT* rshape)
{
    VSLConvTaskPtr task;
    MKL_INT z1 = x1 + y1 - 1, z0 = x0 + y0 -1;
    MKL_INT xshape[2]={x1,x0}, yshape[2]={y1,y0}, zshape[2]={z1,z0};
    double z[zshape[1]*zshape[0] + zshape[0]];
    MKL_INT rank=2;
    int status;

    int mode = VSL_CONV_MODE_AUTO;

    /*
    *  Create task descriptor (create descriptor of problem)
    */
    status = vsldConvNewTask(&task,mode,rank,xshape,yshape,zshape);
    if( status != VSL_STATUS_OK ){
        printf("ERROR: creation of job failed, exit with %d\n", status);
        return 1;
    }

    /*
    *  Execute task (Calculate 2 dimension convolution of two arrays)
    */
    status = vsldConvExec(task,x,NULL,y,NULL,z,NULL);
    if( status != VSL_STATUS_OK ){
        printf("ERROR: job status bad, exit with %d\n", status);
        return 1;
    }

    /*
    *  Delete task object (delete descriptor of problem)
    */
    status = vslConvDeleteTask(&task);
    if( status != VSL_STATUS_OK ){
        printf("ERROR: failed to delete task object, exit with %d\n", status);
        return 1;
    }


    // extract the center vector
    for(int i = 0; i < r1; i++)
        for(int j = 0; j < r0; j++)
        {
            r[i*r0 + j] = z[(i+r1-1)*zshape[0]+ j+r0-1];
        }

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
    double *av=new double[ay], *bv=new double[by], *rv=new double[ry];

    for ( std::size_t x = 0; x < rx; ++x)
        {
            for ( std::size_t i = 0; i < ay; ++i)
                    av[i] = a[x][i][0];
            for ( std::size_t i = 0; i < by; ++i)
                    bv[i] = b[x][i][0];

            // 1d convolution of av and bv, output: rv
            conv_mkl_1d( av, ay, bv, by, rv, ry );

            for ( std::size_t i = 0; i < ry; ++i )
            {
                r[x][i][0] = rv[i];
            }
        }

    // have to free the arrays
    delete[] av, bv, rv;
    return rp;
}

// this 2D convolution is unfinished
// 2D convolution using MKL
inline double3d_ptr bf_conv_mkl_2d(double3d_ptr ap, double3d_ptr bp)
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
    // initialize r
    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
                r[x][y][z] = 0;

    // 2D matrix in yz plane
    double av[az*ay], bv[bz*by], rv[rz*ry];

    for ( std::size_t x = 0; x < rx; ++x)
    {
        for ( std::size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
        {
            // get 2d image of input and kernel
            for (std::size_t y = 0; y < ry; ++y)
                for ( std::size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                    for (std::size_t z = 0; z < rz; ++z)
                        for ( std::size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                        {
                            av[dy*az+dz] = a[dx][dy][dz];
                            bv[dy*bz+wz] = b[wx][wy][wz];
                        }

            // 2d convolution of av and bv, output: rv
            //MKL_INT ashape[2] = {ay,az}, bshape[2]={by,bz}, rshape[2]={ry,rz};
            //conv_sse_2d( av, ashape, bv, bshape, rv, rshape );
            conv_mkl_2d( av, ay, az, bv, by, bz, rv, ry, rz );

            for ( std::size_t y = 0; y < ry; ++y )
                for ( std::size_t z = 0; z < rz; ++z )
                    r[x][y][z] = rv[y*rz+z];
        }
    }

    //delete[] av, bv, rv;
    free(av);
    free(bv);
    free(rv);

    return rp;
}




}} // namespace zi::znn

#endif // ZNN_CONV_SSE_HPP_INCLUDED
