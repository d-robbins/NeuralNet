#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <omp.h>

#include "network.h"

#define NETWORK_SIZE        3

#define INPUT_LAYER_SIZE    2
#define OUTPUT_LAYER_SIZE   2

#define TOPOLOGY         {INPUT_LAYER_SIZE, 12, OUTPUT_LAYER_SIZE}

int main(int argc, char ** argv)
{
    if (argc >= 2)
        omp_set_num_threads(atoi(argv[1]));

    time_t t;
    srand((unsigned)time(&t));

    struct Network nn;

    int topology[NETWORK_SIZE] = TOPOLOGY;

    float input[INPUT_LAYER_SIZE];
    float expected[OUTPUT_LAYER_SIZE];

    int ITERATIONS = 1000;

    nn = create_network(topology, NETWORK_SIZE);

    // train the network
    for (int i = 0; i < ITERATIONS; i++)
    {    
        input[0] = ((i + (rand() % 321)) % 2 == 0) ? 1.0f : 0.0f;
        input[1] = (input[0] == 1.0f) ? 0.0f : 1.0f;

        expected[0] = (input[0] == 1.0f) ? 0.0f : 1.0f;
        expected[1] = (input[1] == 1.0f) ? 0.0f : 1.0f;
        
        feed_input_data(nn, input);

        feed_forward(nn);

        back_propogation(nn, expected);
    }

    // without threshhold (learning rate 0.5): 357 avg 
    // with threshhold (learning rate 0.5): 357 avg
    // with threshhold (learning rate 0.75): 270 avg
    // without threshhold (learning rate 0.75): 269 avg
    // with threshhold (learning rate 0.8): 260 avg
    // with threshhold (learning rate 0.9): 243 avg
    // with threshhold (learning rate 1.0): 228 avg

    print_connections(nn._layers[0]);
    print_connections(nn._layers[1]);

    while(1)
    {
        int i1, i2;

        printf("Enter 2 numbers\n");
        scanf("%d %d", &i1, &i2);

        input[0] = (float)i1;
        input[1] = (float)i2;

        feed_input_data(nn, input);

        feed_forward(nn);

        printf("guess: (%.06f, %.06f)\n",  nn._layers[nn._num_layers-1]._layer_nodes[0]->_value, nn._layers[nn._num_layers-1]._layer_nodes[1]->_value);
    }
    
    free_network(&nn);
    return 0;
}

