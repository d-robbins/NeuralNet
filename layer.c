#include "layer.h"

#include "node.h"

#include <unistd.h>

layer_t create_layer(int size)
{
    layer_t newLayer;
    newLayer._layer_size = size;

    newLayer._layer_nodes = (node_t**)malloc(sizeof(node_t*) * size);

    for (int i = 0; i < size; i++)
    {
        *(newLayer._layer_nodes + i) = create_node();

        // set the nodes activation value (random right now)
        //float value = 1.0f / ((float)((rand() % 10) + 1));
        
        set_node_value(*(newLayer._layer_nodes + i), 0.0f);
    }

    return newLayer;
}

void display_layer_node_values(layer_t layer)
{
    for (int i = 0; i < layer._layer_size; i++)
    {
        printf("%d: %f\n", i, layer._layer_nodes[i]->_value);
    }
}

void attach_layer(layer_t a, layer_t b)
{
    // intialize a layer's node connections
    for (int i = 0; i < a._layer_size; i++)
    {
        a._layer_nodes[i]->_num_connections = b._layer_size;
        a._layer_nodes[i]->_connections = (node_t**)malloc(sizeof(node_t*) * b._layer_size);
        a._layer_nodes[i]->_connection_weights = (float*)malloc(sizeof(float) * b._layer_size);
    }

    // attach each of layer a nodes to each of layer b nodes
    for (int i = 0; i < a._layer_size; i++)
    {
        for (int j = 0; j < b._layer_size; j++)
        {
            a._layer_nodes[i]->_connections[j] = b._layer_nodes[j];

            // set random weights [-5.0f, 5.0f]
            float value = (float)((rand() % 5) + 1) / (float)((rand() % 3) + 1);
            if ((rand() % 100) % 2 == 0)
                value *= -1.0f;

            a._layer_nodes[i]->_connection_weights[j] = value;
        }   
    }
}

void print_connections(layer_t a)
{
    for (int i = 0; i < a._layer_size; i++)
    {
        for (int j = 0; j < a._layer_nodes[i]->_num_connections; j++)
        {
            printf("[%.01f] --%.01f--> [%.01f]\n", a._layer_nodes[i]->_value, a._layer_nodes[i]->_connection_weights[j], a._layer_nodes[i]->_connections[j]->_value);
        }
    }
}

void calculate_activations(layer_t a)
{
    for (int i = 0; i < a._layer_size; i++)
    {
        // run activation function on each layer node
        activation_func(a._layer_nodes[i]);
    } 
}

void propogate(layer_t a)
{
    for (int i = 0; i < a._layer_size; i++)
    {
        // activate each node 
        if (a._layer_nodes[i]->_value > 0.6f)
        {
            activate(a._layer_nodes[i]);
            a._layer_nodes[i]->_activated = true;
        }
        else 
        {
            a._layer_nodes[i]->_activated = false;
        }
    }
}

float* get_activation_values(layer_t a)
{
    float* values = (float*)malloc(sizeof(float) * a._layer_size);
    for (int i = 0; i < a._layer_size; i++)
    {
        values[i] = a._layer_nodes[i]->_value;
    }
    return values;
}

void print_activation_values(layer_t a)
{
    for (int i = 0; i < a._layer_size; i++)
    {
        printf("[%d] %.02f\n", i, a._layer_nodes[i]->_value);
    }
}

void free_layer(layer_t a)
{
    for (int i = 0; i < a._layer_size; i++)
    {
        free (a._layer_nodes[i]);
    }

    free(a._layer_nodes);
}