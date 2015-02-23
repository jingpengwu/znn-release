#!/bin/bash

#g++ -o ./bin/znn src/main.cpp -g src/core/convolve.o -I. -I./src -I./zi -DNDEBUG -O3 -Wall -Wextra -Wno-unused-result -Wno-unused-local-typedefs -lfftw3 -lpthread -lrt -lfftw3_threads -lboost_program_options -lboost_regex -lboost_filesystem -lboost_system -msse2


g++ -c src/main.cpp -g -I. -I./src -I./zi -DNDEBUG -O3 -Wall -Wextra -Wno-unused-result -Wno-unused-local-typedefs -lfftw3 -lpthread -lrt -lfftw3_threads -lboost_program_options -lboost_regex -lboost_filesystem -lboost_system -msse2



cd src/core
gcc -std=c99 -c convolve.c
cd ../../

g++ -o znn -I. -I./src -I./zi -DNDEBUG -O3 -Wall -Wextra -Wno-unused-result -Wno-unused-local-typedefs -lfftw3 -lpthread -lrt -lfftw3_threads -lboost_program_options -lboost_regex -lboost_filesystem -lboost_system -msse2 main.o src/core/convolve.o
#make
