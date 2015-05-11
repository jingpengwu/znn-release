ODIR		=	./lib
CPP			=	g++
CPP_FLAGS	= 	-g -shared -static
INC_FLAGS	=	-I. -I../ -I../../zi -I/usr/people/jingpeng/libs/boost/include  -I/usr/people/jingpeng/libs/anaconda/pkgs/python-2.7.8-1/include/python2.7/
LIB_FLAGS	=	-L. -L../ -L/usr/people/jingpeng/libs/boost/lib64
OPT_FLAGS	=	-DNDEBUG -O3 -march=native
OTH_FLAGS	=	#-Wall -Wextra -Wno-unused-variable
LIBS		=	-lfftw3 -lpthread -lrt -lfftw3_threads
BOOST_LIBS	=	-lboost_program_options -lboost_regex -lboost_filesystem -lboost_system -lboost_timer

pyznn_forward: pyznn.cpp
	$(CPP) -o $(ODIR)/pyznn.so pyznn.cpp $(CPP_FLAGS) $(INC_FLAGS) $(LIB_FLAGS) $(OPT_FLAGS) $(OTH_FLAGS) $(LIBS) $(BOOST_LIBS)


.PHONY: clean

clean:
	rm -f $(ODIR)/*
