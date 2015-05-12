#ifndef ZNN_H_INCLUDED
#define ZNN_H_INCLUDED

#include "../core/network.hpp"
#include "../front_end/options.hpp"
#include <zi/zargs/zargs.hpp>

void pyznn_forward( double*, unsigned int, unsigned int, unsigned int,
                    double*, unsigned int, unsigned int, unsigned int );

#endif // ZNN_H_INCLUDED
