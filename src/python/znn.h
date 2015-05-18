#ifndef ZNN_H_INCLUDED
#define ZNN_H_INCLUDED

#include "../core/network.hpp"
#include "../front_end/options.hpp"


void prepare_test_c( std::string& );
void pyznn_forward_c( double*, unsigned int, unsigned int, unsigned int,
                    double*, unsigned int, unsigned int, unsigned int );

#endif // ZNN_H_INCLUDED
