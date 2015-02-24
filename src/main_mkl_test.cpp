//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
//                     Kisuk Lee           <kisuklee@mit.edu>
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

#include "core/bf_conv.hpp"
#include "core/conv_mkl.hpp"

using namespace zi::znn;

// test MKL performance, added by Jingpeng Wu
bool test_mkl(std::size_t ax, std::size_t ay, std::size_t bx, std::size_t by, int times)
{
    std::cout<< "start testing MKL ..." <<std::endl;
    // initialization
    double3d_ptr ap=volume_pool.get_double3d(ax,ay,1);
    double3d_ptr bp=volume_pool.get_double3d(bx,by,1);
    double3d& a=*ap;
    double3d& b=*bp;
    double3d_ptr rp_m;
    double3d_ptr rp_n;

    // give some value
    for( int i=0; i<ax; ++i )
        for (int j=0; j<ay; ++j)
            a[i][j][0] = i*ay + j ;
    for( int i=0; i<bx; ++i )
        for( int j=0; j<by; ++j )
            b[i][j][0] = i*by + j ;

    // convolution
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // direct convolution with naive method
    boost::timer::cpu_timer timer;
    for(int i=0; i<times; ++i)
        rp_n = bf_conv_original(ap, bp);
    timer.stop();
    // show time
    std::cout <<"time cost of naive method: "<< timer.format() << "s\n"; // gives the number of seconds, as double.

    // convolution with MKL
    timer.start();
    for(int i=0; i<times; ++i)
        rp_m = bf_conv_mkl_2d(ap, bp);
    timer.stop();
    // show timer
    std::cout <<"time cost of MKL method:   "<< timer.format() << "s\n"; // gives the number of seconds, as double.
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // get output volume
    double3d& r_n=*rp_n;
    double3d& r_m=*rp_m;

    // shape of output
    std::cout<< "shape of naive and MKL output: "<<r_n.shape()[0]<<"X"<<r_n.shape()[1]<<",  "<<r_m.shape()[0]<<"X"<<r_m.shape()[1]<<std::endl;
    // test the address
    if (rp_n == rp_m)
        std::cout<< "the address are the same!" << std::endl;

    // test the correctness of MKL result by comparing the results
    bool flag = true;
    for ( int i=0; i<r_n.shape()[0]; ++i)
    {
        for ( int j=0; j<r_n.shape()[1]; ++j )
        {
            if( r_n[i][j][0] != r_m[i][j][0] )
            {
                flag = false;
                std::cout<<"i: "<<i<<",     j: "<<j<<std::endl;
                std::cout<< "the results are different: " << r_n[i][j][0]<< " and " << r_m[i][j][0] << std::endl;
                break;
            }
        }
        //if(!flag)
          //  break;
    }
    if (flag)
        std::cout<< "results are the same." << std::endl;

    // show some value
    std::cout<< "value of naive method  :   " << r_n[0][0][0] << std::endl;
    std::cout<< "value of MKL approach  :   " << r_m[0][0][0] << std::endl;

    return 0;
}

int main(int argc, char** argv)
{
    std::cout<< "test MKL only" << std::endl;
    if (argc == 6)
        test_mkl(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]) );
    else if (argc == 4)
    {
        std::cout<< "parameters :   " << argv[1]<<",    "<<argv[2]<<std::endl;
        std::cout<< "run times  :   " << argv[3]<< std::endl;
        test_mkl(atoi(argv[1]), atoi(argv[1]),atoi(argv[2]), atoi(argv[2]), atoi(argv[3]) );
    }
    else
    {
        std::cout<< "argument should be two or four uint numbers." << std::endl;
        std::cout<< "use default matrix size: 20X20, 5X5"<<std::endl;
        test_mkl(20,20,5,5,10);
    }

    return 0;
}
