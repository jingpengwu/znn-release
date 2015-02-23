
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

//#include "types.hpp"
extern "C" {
    #include "mkl_vsl.h"
}

//namespace zi {
//namespace znn {

bool conv_sse_1d(double* x, MKL_INT xshape, double* y, MKL_INT yshape, double* r, MKL_INT rshape)
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

bool conv_sse_2d(double* x, MKL_INT x1, MKL_INT x0, double* y, MKL_INT y1, MKL_INT y0, double* r, MKL_INT r1, MKL_INT r0)
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


//}} // namespace zi::znn

#endif // ZNN_CONV_SSE_HPP_INCLUDED
