#ifndef __LAYER_H_
#define __LAYER_H_

#include "node.h"

#include <stdio.h>
#include <math.h>

// A -> B

typedef struct Layer
{
    node_t **_layer_nodes;
    int _layer_size;
} layer_t;

layer_t create_layer(int size);

void attach_layer(layer_t a, layer_t b);
void display_layer_node_values(layer_t layer);
void print_connections(layer_t a);

void print_activation_values(layer_t a);
float* get_activation_values(layer_t a);

void calculate_activations(layer_t a);
void propogate(layer_t a);
void free_layer(layer_t a);

#endif // __LAYER_H_