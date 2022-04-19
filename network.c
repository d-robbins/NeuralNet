#include "network.h"

struct Network create_network(int* topology, int nlayers)
{
    struct Network nn;

    nn._num_layers = nlayers;
    nn._layers = (layer_t*)malloc(sizeof(layer_t) * nlayers);

    for (int i = 0; i < nlayers; i++)
    {
        nn._layers[i] = create_layer(topology[i]);

        if (i != 0)
        {
            attach_layer(nn._layers[i-1], nn._layers[i]);
        }
    }

    return nn;
}

void feed_forward(struct Network nn)
{
    for (int i = 0; i < nn._num_layers; i++)
    {
        if (i != 0)
            calculate_activations(nn._layers[i]);

        propogate(nn._layers[i]);
    }
}

void print_network_layer_activations(struct Network nn)
{
    for (int i = 0; i < nn._num_layers; i++)
    {
        printf("-------%d-------\n", i+1);
        print_activation_values(nn._layers[i]);
    }
    printf("----------------\n");
}

void feed_input_data(struct Network nn, float* data)
{
    if (data == NULL)
    {
        perror("input data array null in network.c\n");
        exit(1);
    }

    for (int i = 0; i < nn._layers[0]._layer_size; i++)
    {
        set_node_value(nn._layers[0]._layer_nodes[i], data[i]);
    }
}

float* calculate_error(struct Network nn, float* expected)
{
    if (expected == NULL)
    {
        perror("expected data array null in network.c\n");
        exit(1);
    }

    // get the absolute error of the output network layer
    float* calculated = get_activation_values(nn._layers[nn._num_layers-1]);
    for (int i = 0; i < nn._layers[nn._num_layers-1]._layer_size; i++)
    {
        calculated[i] -= expected[i];
    }

    return calculated;
}

float* calculate_cost_per_node(struct Network nn, float* expected)
{
    float* calculated = get_activation_values(nn._layers[nn._num_layers-1]);
    for (int i = 0; i < nn._layers[nn._num_layers-1]._layer_size; i++)
    {
        calculated[i] = powf(calculated[i] - expected[i], 2.0f);
    }

    return calculated;
}

void free_network(struct Network* network)
{
    for (int i = 0; i < network->_num_layers; i++)
    {
        free_layer(network->_layers[i]);
    }

    free(network->_layers);
}

void back_propogation(struct Network nn, float* expected)
{
    float learningRate = 1.0f;

    // loop through each node of the output layer
    for (int j = 0; j < nn._layers[nn._num_layers-1]._layer_size; j++)
    {
        // loop through each node of the previous layer
        for (int k = 0; k < nn._layers[nn._num_layers-2]._layer_size; k++)
        {
            // weight from node k in previous layer to node j in output layer
            // float wkj = nn._layers[nn._num_layers-2]._layer_nodes[k]->_connection_weights[j];

            // activation value of node k
            float yk = nn._layers[nn._num_layers-2]._layer_nodes[k]->_value;

            float sigderiv = nn._layers[nn._num_layers-1]._layer_nodes[j]->_value * 
                    (1.0f - nn._layers[nn._num_layers-1]._layer_nodes[j]->_value);

            // because j is the output layer

            float perr = -1.0f * (expected[j] - nn._layers[nn._num_layers-1]._layer_nodes[j]->_value);

            float errterm = -1.0f * sigderiv * perr;

            float weight_gradient = errterm * learningRate * yk;

            // printf("weight for node %d to node %d: %0.2f\n", k, j, nn._layers[nn._num_layers-2]._layer_nodes[k]->_connection_weights[j]);
            // printf("weight gradient for node %d to node %d: %0.2f\n", k, j, weight_gradient);

            nn._layers[nn._num_layers-2]._layer_nodes[k]->_connection_weights[j] += weight_gradient;
        }
    }
}