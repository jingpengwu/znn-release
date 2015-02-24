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


using namespace zi::znn;

int main(int argc, char** argv)
{
    std::cout<< "test MKL only" << std::endl;
    if (argc == 5)
        test_mkl(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
    else if (argc == 3)
    {
        std::cout<< "parameters:" << argv[1]<<",    "<<argv[2]<<std::endl;
        test_mkl(atoi(argv[1]), atoi(argv[1]),atoi(argv[2]), atoi(argv[2]));
    }
    else
    {
        std::cout<< "argument should be two or four uint numbers." << std::endl;
        std::cout<< "use default matrix size: 20X20, 5X5"<<std::endl;
        test_mkl(20,20,5,5);
    }
    return 0;
}
