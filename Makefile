ODIR		=	./bin
CPP		=	g++
CPP_FLAGS	= 	-g
INC_FLAGS	=	-I. -I./src -I./zi -I/usr/local/boost/1.55.0/include 
LIB_FLAGS	=	-L. -L/usr/local/boost/1.55.0/lib64 
OPT_FLAGS	=	-DNDEBUG -O3 -march=native
OTH_FLAGS	=	-Wall -Wextra -Wno-unused-variable

LIBS		=	-lfftw3 -lpthread -lrt -lfftw3_threads
BOOST_LIBS	=	-lboost_program_options -lboost_regex -lboost_filesystem -lboost_system -lboost_timer

znn: src/main.cpp
	$(CPP) -o $(ODIR)/znn src/main.cpp $(CPP_FLAGS) $(INC_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS) $(BOOST_LIBS) $(SSE_FLAGS)


# intel MKL 
CC		=	icpc
CC_FLAGS	=	-O3 #-xHost #-ipo -static-intel
DEBUG_FLAGS	=	-g -debug
INC_MKL_FLAGS	=	-I. -I./src -I./zi -I/usr/local/boost/1.55.0/include -I/opt/intel/composer_xe_2013_sp1.4.211/mkl/include/
LIB_MKL_FLAGS	=	-L/opt/intel/composer_xe_2013_sp1.4.211/mkl/lib/intel64/ -L/opt/intel/composer_xe_2013_sp1.4.211/compiler/lib/intel64 -L. -L/usr/local/boost/1.55.0/lib64 
MKL_FLAGS	=	-lmkl_core -lm -lmkl_intel_lp64 -lmkl_sequential

.PHONY: mkl mkl_test clean

mkl:
	$(CC) -o $(ODIR)/znn_mkl src/main.cpp $(CC_FLAGS) $(INC_MKL_FLAGS) $(LIB_MKL_FLAGS) $(OTH_FLAGS) $(LIBS) $(MKL_FLAGS) $(BOOST_LIBS) 

mkl_test:
	$(CC) -o $(ODIR)/mkl_test src/main_mkl_test.cpp $(CC_FLAGS) $(INC_MKL_FLAGS) $(LIB_MKL_FLAGS) $(OTH_FLAGS) $(LIBS) $(MKL_FLAGS) $(BOOST_LIBS) 

clean:
	rm -f $(ODIR)/*
