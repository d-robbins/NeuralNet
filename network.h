#ifndef __NETWORK_H_
#define __NETWORK_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

struct WeightMatrix
{
    int _r;
    int _c;
    float **_weights;
};

struct Node
{
    float _activation;
    float _errterm;
};

struct Network
{
    int _num_layers;

    struct Node ** _avals;
    struct WeightMatrix* _wmatrix;
    int * _top;
};

struct Network create_network(int*, int);

void feed_forward(struct Network);
void back_propagation(struct Network, float*);

void initialize_weight_matrix(struct WeightMatrix*);

void print_weight_matrices(struct Network);
void print_activations(struct Network);

void feed_input_data(struct Network, float*);

void free_network(struct Network*);

void write_weight_images(struct Network);
void write_weight_image(struct Network, int);

#endif // __NETWORK_H_