//
// Copyright (C) 2014  Kisuk Lee <kisuklee@mit.edu>
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

#ifndef ZNN_FORWARD_LOGISTIC_ERROR_FN_HPP_INCLUDED
#define ZNN_FORWARD_LOGISTIC_ERROR_FN_HPP_INCLUDED

#include "error_fn.hpp"
#include "../core/volume_pool.hpp"

namespace zi {
namespace znn {

class forward_logistic_error_fn: virtual public logistic_error_fn
{
public:
    virtual double3d_ptr gradient(double3d_ptr dEdF, double3d_ptr F)
    {
        double3d_ptr r = volume_pool.get_double3d(F);
        (*r) = (*dEdF);
        return r;
    }

}; // class forward_logistic_error_fn

}} // namespace zi::znn

#endif // ZNN_FORWARD_LOGISTIC_ERROR_FN_HPP_INCLUDED
