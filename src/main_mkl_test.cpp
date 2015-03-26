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
#include <boost/random.hpp>

using namespace zi::znn;

double3d_ptr random_volume(double3d_ptr ap, float seed)
{
    // Initialise Boost random numbers, uniform integers from min to max
    const int rangeMin = 0;
    const int rangeMax = 10;
    typedef boost::uniform_int<> NumberDistribution; // use int for evaluation. real may cause inequality due to precision problems
    typedef boost::mt19937 RandomNumberGenerator;
    typedef boost::variate_generator< RandomNumberGenerator&, NumberDistribution > Generator;

    NumberDistribution distribution( rangeMin, rangeMax );
    RandomNumberGenerator generator;
    Generator numberGenerator(generator, distribution);
    generator.seed( seed ); // seed with some initial value

    double3d& a=*ap;
    for (std::size_t i=0; i < a.shape()[0]; i++)
    for(std::size_t j=0; j < a.shape()[1]; j++)
        for( std::size_t k = 0; k < a.shape()[2]; k++ )
            a[i][j][k] = numberGenerator();
    return ap;
}

void assert_volume(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a=*ap;
    double3d& b=*bp;

    // shape of output
    std::cout<< "shape of naive and MKL output: "<<a.shape()[0]<<"X"<<a.shape()[1]<<",  "<<b.shape()[0]<<"X"<<b.shape()[1]<<std::endl;

    // test the correctness of MKL result by comparing the results
    bool flag = true;
    for ( int i=0; i<a.shape()[0]; ++i)
    {
        for ( int j=0; j<a.shape()[1]; ++j )
        {
            for ( int k=0; k<a.shape()[2]; ++k )
            {
                if( a[i][j][k] != b[i][j][k] )
                {
                    flag = false;
                    std::cout<<"i: "<< i <<",   j: "<< j <<",   k: "<< k <<std::endl;
                    std::cout<< "naive: " << a[i][j][k]<< " and MKL:" << b[i][j][k] << std::endl;
                    break;
                }
            }
            if(!flag)
                break;
        }
        if(!flag)
            break;
    }

    if (flag)
        std::cout<< "results are the same." << std::endl;
}

// test MKL performance, added by Jingpeng Wu
bool test_mkl(vec3i ashape, vec3i bshape, vec3i s, int times)
{
    std::cout<< "start testing MKL ..." <<std::endl;
    // initialization
    double3d_ptr ap=volume_pool.get_double3d( ashape );
    double3d_ptr bp=volume_pool.get_double3d( bshape );
    double3d& a=*ap;
    double3d& b=*bp;

    double3d_ptr rp_n;
    double3d_ptr rp_m;

    // setup timer
    boost::timer::cpu_timer n_timer;
    n_timer.start();
    n_timer.stop();
    boost::timer::cpu_timer m_timer;
    m_timer.start();
    m_timer.stop();

    // convolution
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    for(int i=0; i<times; ++i)
    {
        // randomize volume
        ap = random_volume(ap, i);
        bp = random_volume(bp, i+2);

        // convolution using naive method
        n_timer.resume();
        //rp_n = bf_conv_naive(ap, bp);
        rp_n = bf_conv_sparse_naive(ap, bp, s);
        n_timer.stop();

        // convolution using MKL
        m_timer.resume();
        //rp_m = bf_conv_mkl(ap, bp);
        rp_m = bf_conv_sparse_mkl(ap, bp, s);
        m_timer.stop();
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // show time
    std::cout <<"time cost of naive method: "<< n_timer.format() << std::endl; // gives the number of seconds, as double.
    std::cout <<"time cost of MKL method:   "<< m_timer.format() << std::endl; // gives the number of seconds, as double

    //save volume for further examination
    volume_utils::save(rp_n, "output_naive.image");
    volume_utils::save(rp_m, "output_mkl.image");

    // assert the output of naive and MKL method is the same
    assert_volume( rp_n, rp_m );

    return 0;
}

int main(int argc, char** argv)
{
    std::cout<< "test MKL only" << std::endl;

    if (argc == 11)
    {
        // get the shape
        vec3i ashape( atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) );
        vec3i bshape( atoi(argv[4]), atoi(argv[5]), atoi(argv[6]) );
        vec3i s( atoi(argv[7]), atoi(argv[8]), atoi(argv[9]) );
        int times = atoi( argv[10] );
        test_mkl( ashape, bshape, s, times );
    }
    else
    {
        std::cout<< "argument should be 7 uint numbers." << std::endl;
        std::cout<< "use default matrix size: 10X10X1, 5X5X1, "<<std::endl;
        vec3i ashape(10,10,1);
        vec3i bshape(5,5,1);
        vec3i s(2,2,1);
        int times = 1;
        test_mkl( ashape, bshape, s, times );
    }

    return 0;
}
