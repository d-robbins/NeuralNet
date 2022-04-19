#include "node.h"

node_t* create_node()
{
    node_t* newNode = (node_t*)malloc(sizeof(node_t));

    *newNode = (node_t){._connection_weights = NULL, ._connections = NULL, ._value = 0.0f, ._num_connections = 0, ._bias = 0.0f, ._activated = false};

    return newNode;
}

void set_node_value(node_t* node, float value)
{
    node->_value = value;
}

void assign_node_weights(node_t* node, float* weights)
{   
    if (node->_connection_weights == NULL)
    {
        perror("connection weights array null\n");
        exit(1);
        return;
    }

    for (int i = 0; i < node->_num_connections; i++)
    {
        node->_connection_weights[i] = (weights == NULL) ? 1.0f : weights[i];
    }
}

void activate(node_t* node)
{
    if (node->_num_connections == 0)
        return;

    for (int i = 0; i < node->_num_connections; i++)
    {
        if (node->_connections[i] == NULL)
        {
            fprintf(stderr, "connection %i of node with value %0.2f null", i, node->_value);
            exit(1);
        }

        node->_connections[i]->_value += node->_connection_weights[i] * node->_value;
    }
}

void activation_func(node_t* node)
{
    // sigmoid
    node->_value += node->_bias;

    float ex = expf(-1.0f * node->_value) + 1.0f;
    node->_value = 1.0f / ex;
}
