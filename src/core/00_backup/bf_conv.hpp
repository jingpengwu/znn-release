//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_BF_CONV_HPP_INCLUDED
#define ZNN_BF_CONV_HPP_INCLUDED

#include "types.hpp"
#include "volume_pool.hpp"
//#include <smmintrin.h> // SSE4
//#include <immintrin.h> // AVX
#include <emmintrin.h> // SSE2
//#include "conv_sse.hpp"
//#include "convolve.h"
//extern "C" {
  //  #include "convolve.h"
//}

namespace zi {
namespace znn {

/* Vectorize the algorithm to compute 4 output samples in parallel.
 *
 * Each kernel value is repeated 4 times, which can then be used on
 * 4 input samples in parallel. Stepping over these as in naive
 * means that we get 4 output samples for each inner kernel loop.
 *
 * For this, we need to pre-reverse the kernel, rather than doing
 * the loopup each time in the inner loop.
 *
 * The last value needs to be done as a special case.
 */
int convolve_sse_simple(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[4] __attribute__ ((aligned (16)));

    __m128 kernel_reverse[kernel_length] __attribute__ ((aligned (16)));
    __m128 data_block __attribute__ ((aligned (16)));

    __m128 prod __attribute__ ((aligned (16)));
    __m128 acc __attribute__ ((aligned (16)));

    // Reverse the kernel and repeat each value across a 4-vector
    for(int i=0; i<kernel_length; i++){
        kernel_block[0] = kernel[kernel_length - i - 1];
        kernel_block[1] = kernel[kernel_length - i - 1];
        kernel_block[2] = kernel[kernel_length - i - 1];
        kernel_block[3] = kernel[kernel_length - i - 1];

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    for(int i=0; i<length-kernel_length; i+=4){

        // Zero the accumulator
        acc = _mm_setzero_ps();

        /* After this loop, we have computed 4 output samples
         * for the price of one.
         * */
        for(int k=0; k<kernel_length; k++){

            // Load 4-float data block. These needs to be an unaliged
            // load (_mm_loadu_ps) as we step one sample at a time.
            data_block = _mm_loadu_ps(in + i + k);
            prod = _mm_mul_ps(kernel_reverse[k], data_block);

            // Accumulate the 4 parallel values
            acc = _mm_add_ps(acc, prod);
        }
        _mm_storeu_ps(out+i, acc);

    }

    // Need to do the last value as a special case
    int i = length - kernel_length;
    out[i] = 0.0;
    for(int k=0; k<kernel_length; k++){
        out[i] += in[i+k] * kernel[kernel_length - k - 1];
    }

    return 0;
}


/* As convolve_sse_simple plus...
 *
 * We specify that the kernel must have a length which is a multiple
 * of 4. This allows us to define a fixed inner-most loop that can be
 * unrolled by the compiler
 */
int convolve_sse_partial_unroll(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[4] __attribute__ ((aligned (16)));

    __m128 kernel_reverse[kernel_length] __attribute__ ((aligned (16)));
    __m128 data_block __attribute__ ((aligned (16)));

    __m128 prod __attribute__ ((aligned (16)));
    __m128 acc __attribute__ ((aligned (16)));

    // Repeat the kernel across the vector
    for(int i=0; i<kernel_length; i++){
        kernel_block[0] = kernel[kernel_length - i - 1];
        kernel_block[1] = kernel[kernel_length - i - 1];
        kernel_block[2] = kernel[kernel_length - i - 1];
        kernel_block[3] = kernel[kernel_length - i - 1];

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    for(int i=0; i<length-kernel_length; i+=4){

        acc = _mm_setzero_ps();

        for(int k=0; k<kernel_length; k+=4){

            int data_offset = i + k;

            for (int l = 0; l < 4; l++){

                data_block = _mm_loadu_ps(in + data_offset + l);
                prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                acc = _mm_add_ps(acc, prod);
            }
        }
        _mm_storeu_ps(out+i, acc);

    }

    // Need to do the last value as a special case
    int i = length - kernel_length;
    out[i] = 0.0;
    for(int k=0; k<kernel_length; k++){
        out[i] += in[i+k] * kernel[kernel_length - k - 1];
    }

    return 0;
}


/* As convolve_sse_partial_unroll plus...
 *
 * We repeat the input data 4 times, with each repeat being shifted
 * by one sample from the previous repeat:
 * original: [0, 1, 2, 3, 4, 5, ...]
 *
 * repeat 1: [0, 1, 2, 3, 4, 5, ...]
 * repeat 2: [1, 2, 3, 4, 5, 6, ...]
 * repeat 3: [2, 3, 4, 5, 6, 7, ...]
 * repeat 4: [3, 4, 5, 6, 7, 8, ...]
 *
 * The effect of this is to create a set of arrays that encapsulate
 * a 16-byte alignment for every possible offset within the data.
 * Sample 0 is aligned in repeat 1, Sample 1 is aligned in repeat 1
 * etc. We then wrap around and sample 4 is aligned on repeat 1.
 *
 * The copies can be done fast with a memcpy.
 *
 * This means that in our unrolled inner-most loop, we can now do
 * an aligned data load (_mm_load_ps), speeding up the algorithm
 * by ~2x.
 * */
int convolve_sse_in_aligned(float* in, float* out, int length,
        float* kernel, int kernel_length)
{
    float kernel_block[4] __attribute__ ((aligned (16)));
    float in_aligned[4][length] __attribute__ ((aligned (16)));

    __m128 kernel_reverse[kernel_length] __attribute__ ((aligned (16)));
    __m128 data_block __attribute__ ((aligned (16)));

    __m128 prod __attribute__ ((aligned (16)));
    __m128 acc __attribute__ ((aligned (16)));

    // Repeat the kernel across the vector
    for(int i=0; i<kernel_length; i++){
        kernel_block[0] = kernel[kernel_length - i - 1];
        kernel_block[1] = kernel[kernel_length - i - 1];
        kernel_block[2] = kernel[kernel_length - i - 1];
        kernel_block[3] = kernel[kernel_length - i - 1];

        kernel_reverse[i] = _mm_load_ps(kernel_block);
    }

    /* Create a set of 4 aligned arrays
     * Each array is offset by one sample from the one before
     */
    for(int i=0; i<4; i++){
        memcpy(in_aligned[i], (in+i), (length-i)*sizeof(float));
    }

    for(int i=0; i<length-kernel_length; i+=4){

        acc = _mm_setzero_ps();

        for(int k=0; k<kernel_length; k+=4){

            int data_offset = i + k;

            for (int l = 0; l < 4; l++){

                data_block = _mm_load_ps(in_aligned[l] + data_offset);
                prod = _mm_mul_ps(kernel_reverse[k+l], data_block);

                acc = _mm_add_ps(acc, prod);
            }
        }
        _mm_storeu_ps(out+i, acc);

    }

    // Need to do the last value as a special case
    int i = length - kernel_length;
    out[i] = 0.0;
    for(int k=0; k<kernel_length; k++){
        out[i] += in_aligned[0][i+k] * kernel[kernel_length - k - 1];
    }

    return 0;
}

inline double3d_ptr bf_conv_V1(double3d_ptr ap, double3d_ptr bp)
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

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
                    for ( std::size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                        for ( std::size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                        {
                            r[x][y][z] +=
                                a[dx][dy][dz] *
                                b[wx][wy][wz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv(double3d_ptr ap, double3d_ptr bp)
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

    // one dimension vector in z direction for SSE, vector and window
    /*double3d_ptr avp = volume_pool.get_double3d(1,1,az);
    double3d_ptr bvp = volume_pool.get_double3d(1,1,bz);
    double3d_ptr rvp = volume_pool.get_double3d(1,1,rz);
    double3d& av = *avp, bv = *bvp, rv = *rvp;*/
    float* av = (float *) malloc( sizeof(float) * az );
    float* bv = (float *) malloc( sizeof(float) * bz );
    float* rv = (float *) malloc( sizeof(float) * rz );
    //float* av = new float[ sizeof(float) * az ];
    //float* bv = new float[ sizeof(float) * bz ];
    //float* rv = new float[ sizeof(float) * rz ];

    for ( std::size_t x = 0; x < rx; ++x)
        for ( std::size_t y = 0; y < ry; ++y )
        {
            for (std::size_t z = 0; z < rz; ++z)
                rv[z] = float( r[x][y][z] );
            for ( std::size_t dx = x, wx = bx - 1; dx < bx + x; ++dx, --wx )
                for ( std::size_t dy = y, wy = by - 1; dy < by + y; ++dy, --wy )
                {
                    for (std::size_t z = 0; z < rz; ++z)
                        for ( std::size_t dz = z, wz = bz - 1; dz < bz + z; ++dz, --wz )
                        {
                            av[dz] = float( a[dx][dy][dz] );
                            bv[wz] = float( b[wx][wy][wz] );
                        }

                    // 1d convolution of av and bv, output: rv
                    //conv_sse_1d( av, az, bv, bz, rv, rz );
                    convolve_sse_simple( av, rv, rz, bv, bz );
                    //convolve_sse_partial_unroll( av, rv, rz, bv, bz );
                    //convolve_sse_in_aligned( av, rv, rz, bv, bz );
                    //convolve_sse_partial_unroll( av, rv, rz, bv, bz );


                    for ( std::size_t z = 0; z < rz; ++z )
                    {
                        r[x][y][z] = double( rv[z] );
                    }
                }
        }

    //delete[] av, bv, rv;
    return rp;
}





inline double3d_ptr bf_conv_old(double3d_ptr ap, double3d_ptr bp)
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

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = 0; dx < bx; ++dx )
                    for ( std::size_t dy = 0; dy < by; ++dy )
                        for ( std::size_t dz = 0; dz < bz; ++dz )
                        {
                            r[x][y][z] +=
                                a[x+dx][y+dy][z+dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_flipped(double3d_ptr ap, double3d_ptr bp)
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

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = 0; dx < bx; ++dx )
                    for ( std::size_t dy = 0; dy < by; ++dy )
                        for ( std::size_t dz = 0; dz < bz; ++dz )
                        {
                            r[x][y][z] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_inverse(double3d_ptr ap, double3d_ptr bp)
{
    double3d& a = *ap;
    double3d& b = *bp;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    std::size_t bx = b.shape()[0];
    std::size_t by = b.shape()[1];
    std::size_t bz = b.shape()[2];

    std::size_t rx = ax + bx - 1;
    std::size_t ry = ay + by - 1;
    std::size_t rz = az + bz - 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    std::fill_n(r.data(), rx*ry*rz, 0);

    for ( std::size_t dx = 0; dx < bx; ++dx )
        for ( std::size_t dy = 0; dy < by; ++dy )
            for ( std::size_t dz = 0; dz < bz; ++dz )
            {
                std::size_t fx = bx - 1 - dx;
                std::size_t fy = by - 1 - dy;
                std::size_t fz = bz - 1 - dz;

                for ( std::size_t x = 0; x < ax; ++x )
                    for ( std::size_t y = 0; y < ay; ++y )
                        for ( std::size_t z = 0; z < az; ++z )
                        {
                            r[x+fx][y+fy][z+fz] += a[x][y][z] * b[dx][dy][dz];
                        }
            }
    return rp;
}

inline double3d_ptr bf_conv_constant(const double3d_ptr& ap,
                                           double b)
{
    double3d& a = *ap;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    double3d_ptr rp = volume_pool.get_double3d(ax,ay,az);
    double3d& r = *rp;

    for ( std::size_t x = 0; x < ax; ++x )
        for ( std::size_t y = 0; y < ay; ++y )
            for ( std::size_t z = 0; z < az; ++z )
            {
                r[x][y][z] = a[x][y][z] * b;
            }

    return rp;
}

inline double bf_conv_flipped_constant(const double3d_ptr& ap,
                                       const double3d_ptr& bp)
{
    ASSERT_SAME_SIZE(ap,bp);

    std::size_t n = ap->num_elements();

    double r = 0;

    double3d& a = *ap;
    double3d& b = *bp;

    for ( std::size_t i = 0; i < n; ++i )
    {
        r += a.data()[i] * b.data()[i];
    }

    return r;
}

inline double3d_ptr bf_conv_inverse_constant(const double3d_ptr& ap,
                                                   double b)
{
    double3d& a = *ap;

    std::size_t ax = a.shape()[0];
    std::size_t ay = a.shape()[1];
    std::size_t az = a.shape()[2];

    double3d_ptr rp = volume_pool.get_double3d(ax,ay,az);
    double3d& r = *rp;

    std::size_t n = ap->num_elements();

    for ( std::size_t i = 0; i < n; ++i )
    {
        r.data()[i] = a.data()[i] * b;
    }

    return rp;
}


inline double3d_ptr bf_conv_sparse(const double3d_ptr& ap,
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

    for ( std::size_t x = 0; x < rx; ++x )
        for ( std::size_t y = 0; y < ry; ++y )
            for ( std::size_t z = 0; z < rz; ++z )
            {
                r[x][y][z] = 0;
             
                for ( std::size_t dx = x, wx = bx-1; dx < bx + x; dx += s[0], wx -= s[0] )
                    for ( std::size_t dy = y, wy = by-1; dy < by + y; dy += s[1], wy -= s[1] )
                        for ( std::size_t dz = z, wz = bz-1; dz < bz + z; dz += s[2], wz -= s[2] )
                        {
                            r[x][y][z] +=
                                a[dx][dy][dz] *
                                b[wx][wy][wz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_flipped_sparse(const double3d_ptr& ap,
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

    for ( std::size_t x = 0; x < rx; x += s[0])
        for ( std::size_t y = 0; y < ry; y += s[1] )
            for ( std::size_t z = 0; z < rz; z += s[2] )
            {
                r[x][y][z] = 0;
                for ( std::size_t dx = 0; dx < bx; ++dx )
                    for ( std::size_t dy = 0; dy < by; ++dy )
                        for ( std::size_t dz = 0; dz < bz; ++dz )
                        {
                            r[x][y][z] +=
                                a[ax-1-x-dx][ay-1-y-dy][az-1-z-dz] *
                                b[bx-1-dx][by-1-dy][bz-1-dz];
                        }
            }

    return rp;
}

inline double3d_ptr bf_conv_inverse_sparse(const double3d_ptr& ap,
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

    std::size_t rx = ax + bx - 1;
    std::size_t ry = ay + by - 1;
    std::size_t rz = az + bz - 1;

    double3d_ptr rp = volume_pool.get_double3d(rx,ry,rz);
    double3d& r = *rp;

    std::fill_n(r.data(), rx*ry*rz, 0);

    for ( std::size_t dx = 0; dx < bx; dx += s[0] )
        for ( std::size_t dy = 0; dy < by; dy += s[1])
            for ( std::size_t dz = 0; dz < bz; dz += s[2] )
            {
                std::size_t fx = bx - 1 - dx;
                std::size_t fy = by - 1 - dy;
                std::size_t fz = bz - 1 - dz;

                for ( std::size_t x = 0; x < ax; ++x )
                    for ( std::size_t y = 0; y < ay; ++y )
                        for ( std::size_t z = 0; z < az; ++z )
                        {
                            r[x+fx][y+fy][z+fz] += a[x][y][z] * b[dx][dy][dz];
                        }
            }
    return rp;
}

}} // namespace zi::znn

#endif // ZNN_BF_CONV_HPP_INCLUDED
