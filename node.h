#ifndef __NODE_H_
#define __NODE_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

typedef struct Node
{
    int _num_connections;
    float _value;
    float _bias;
    bool _activated;
    float _errterm;
    
    float *_connection_weights;
    struct Node **_connections;
} node_t;

node_t* create_node();
void set_node_value(node_t* node, float value);
void assign_node_weights(node_t* node, float* weights);
void activate(node_t* node);
void activation_func(node_t* node);

#endif // __NODE_H_