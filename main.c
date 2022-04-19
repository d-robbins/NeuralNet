#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#include "network.h"

#define NETWORK_SIZE        2

#define INPUT_LAYER_SIZE    2
#define OUTPUT_LAYER_SIZE   2

#define TOPOLOGY         {INPUT_LAYER_SIZE, OUTPUT_LAYER_SIZE}

int main()
{
    time_t t;
    srand((unsigned)time(&t));

    struct Network nn;

    int topology[NETWORK_SIZE] = TOPOLOGY;

    float input[INPUT_LAYER_SIZE];
    float expected[OUTPUT_LAYER_SIZE];

    float avg = 0.0f;
    int ITERATIONS = 1;

    // without threshhold (learning rate 0.5): 357 avg 
    // with threshhold (learning rate 0.5): 357 avg
    // with threshhold (learning rate 0.75): 270 avg
    // without threshhold (learning rate 0.75): 269 avg
    // with threshhold (learning rate 0.8): 260 avg
    // with threshhold (learning rate 0.9): 243 avg
    // with threshhold (learning rate 1.0): 228 avg

    for (int run = 0; run < ITERATIONS; run++)
    {
        nn = create_network(topology, NETWORK_SIZE);

        int NUM_TRAINING = 100000;
        int count = 0;
        for (int i = 0; i < NUM_TRAINING; i++)
        {    
            input[0] = ((i + (rand() % 321)) % 2 == 0) ? 1.0f : 0.0f;
            input[1] = (input[0] == 1.0f) ? 0.0f : 1.0f;

            expected[0] = (input[0] == 1.0f) ? 0.0f : 1.0f;
            expected[1] = (input[1] == 1.0f) ? 0.0f : 1.0f;
            
            // if (i < 10 || i > (NUM_TRAINING - 10))
            //     printf("problem %d: (%0.2f, %0.2f)\n", i, input[0], input[1]);

            feed_input_data(nn, input);

            feed_forward(nn);

            // if (i < 10 || i > (NUM_TRAINING - 10))
            //     printf("guess %i: (%0.2f, %0.2f)\n", i, nn._layers[1]._layer_nodes[0]->_value, nn._layers[1]._layer_nodes[1]->_value);

            float* err = calculate_error(nn, expected);

            if (fabsf(err[0]) < 0.05 && fabsf(err[1]) < 0.05)
            {
                count++;
            }
            else
            {
                count = 0;
            }

            if (count >= 10000)
            {
                avg += (float)i;
                break;
            }

            back_propogation(nn, expected);
        }

        //free_network(&nn);
    }

    printf("avg training time (%d iterations) %0.0f\n", ITERATIONS, avg / (float)ITERATIONS);    
    
    print_connections(nn._layers[0]);

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