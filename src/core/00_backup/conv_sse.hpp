
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
//#include <smmintrin.h> // SSE4
#include <immintrin.h> // SSE3
extern "C" {
    #include "convolve.h"
}

#define SSE3

namespace zi {
namespace znn {

bool conv_sse_1d(float* fav, int la, float* fbv, int lb, float* frv, int lr)
{
    // convolution using SSE
    convolve_sse_simple( fav, frv, lr, fbv, lb );

    return true;
}

}} // namespace zi::znn

#endif // ZNN_CONV_SSE_HPP_INCLUDED
