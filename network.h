#ifndef __NETWORK_H_
#define __NETWORK_H_

#include "layer.h"

struct Network
{
    layer_t* _layers;
    int _num_layers;
};

struct Network create_network(int*, int);

void feed_forward(struct Network);
void back_propogation(struct Network, float*);

void print_network_layer_activations(struct Network);

void feed_input_data(struct Network, float*);
float* calculate_error(struct Network, float*);
float* calculate_cost_per_node(struct Network, float*);

void free_network(struct Network*);

#endif // __NETWORK_H_