#include "network.h"
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct Network create_network(int* topology, int nlayers)
{
    struct Network nn;

    nn._wmatrix = (struct WeightMatrix*)malloc(sizeof(struct WeightMatrix) * nlayers - 1);

    for (int i = 1; i < nlayers; i++)
    {
        nn._wmatrix[i - 1]._r = topology[i];
        nn._wmatrix[i - 1]._c = topology[i - 1];

        initialize_weight_matrix(&nn._wmatrix[i-1]);
    }

    nn._num_layers = nlayers;  
    nn._top = topology;

    nn._avals = (struct Node**)malloc(sizeof(struct Node*) * nlayers);
    for (int i = 0; i < nlayers; i++)
    {
        nn._avals[i] = (struct Node*)malloc(sizeof(struct Node) * topology[i]);
        for (int j = 0; j < topology[i]; j++)
        {
            float value = (float)((rand() % 5) + 1) / (float)((rand() % 3) + 1);
            if ((rand() % 100) % 2 == 0)
                value *= -1.0f;
            nn._avals[i][j]._activation = value;
            nn._avals[i][j]._errterm = 1.0f;
        }
    }

    return nn;
}

void print_activations(struct Network nn)
{
    for (int i = 0; i < nn._num_layers; i++)
    {
        printf("\n-------------Layer %d---------------\n", i);
        for (int j = 0; j < nn._top[i]; j++)
        {
            printf("%d: %.2f\tError Term: %.2f\n", j, nn._avals[i][j]._activation, nn._avals[i][j]._errterm);
        }
        printf("\n---------------------------\n");
    }
}

void print_weight_matrices(struct Network nn)
{
    printf("Weight Matrices\n----------------------------\n");
    for (int i = 0; i < nn._num_layers - 1; i++)
    {
        for (int j = 0; j < nn._wmatrix[i]._r; j++)
        {
            for (int k = 0; k < nn._wmatrix[i]._c; k++)
            {
                printf("%.2f ", nn._wmatrix[i]._weights[j][k]);
            }
            printf("\n");
        }
        printf("---------------------------\n");
    }
}

void initialize_weight_matrix(struct WeightMatrix* wm)
{
    wm->_weights = (float**)malloc(sizeof(float*) * wm->_r);
    for (int i = 0; i < wm->_r; i++)
    {
        wm->_weights[i] = (float*)malloc(sizeof(float) * wm->_c);
        for (int j = 0; j < wm->_c; j++)
        {
            float value = (float)((rand() % 5) + 1) / (float)((rand() % 3) + 1);
            if ((rand() % 100) % 2 == 0)
                value *= -1.0f;
            wm->_weights[i][j] = value;
        }
    }
}

void feed_forward(struct Network nn)
{
    for (int i = 0; i < nn._num_layers - 1; i++)
    {
        struct Node * x = nn._avals[i];
        float * tmp = (float*)malloc(sizeof(float) * nn._top[i + 1]);
        for (int j = 0; j < nn._top[i + 1]; j++)
        {
            tmp[j] = 0;
        }

        for (int j = 0; j < nn._wmatrix[i]._r; j++)
        {
            for (int k = 0; k < nn._wmatrix[i]._c; k++)
            {
                tmp[j] += nn._wmatrix[i]._weights[j][k] * x[k]._activation;
            }
        }

        for (int j = 0; j < nn._top[i + 1]; j++)
        {   
            nn._avals[i + 1][j]._activation = 1.0f / (expf(-1.0f * tmp[j]) + 1.0f);
        }     

        free(tmp);
    }
}

void feed_input_data(struct Network nn, float* data)
{
    if (data == NULL)
    {
        perror("input data array null in network.c\n");
        exit(1);
    }

    for (int i = 0; i < nn._top[0]; i++)
    {
        nn._avals[0][i]._activation = data[i];
    }
}

void free_network(struct Network* network)
{
    for (int i = 0; i < network->_num_layers; i++)
    {
        free(network->_avals[i]);
    }

    free(network->_avals);

    for (int i = 0; i < network->_num_layers - 1; i++)
    {
        for (int j = 0; j < network->_wmatrix[i]._r; j++)
        {
            free(network->_wmatrix[i]._weights[j]);
        }

        free(network->_wmatrix[i]._weights);
    }

    free(network->_wmatrix);
}

void back_propagation(struct Network nn, float* expected)
{
    float learningRate = 1.0f;
    float y_k, y_j, sigderiv, perr, errterm, weight_gradient;

    for (int i = nn._num_layers - 1; i > 0; i--)
    {
        int current_layer = i;
        int next_layer = i + 1;
        int previous_layer = i - 1;

        for (int j = 0; j < nn._top[current_layer]; j++)
        {
            for (int k = 0; k < nn._top[previous_layer]; k++)
            {
                y_k = nn._avals[previous_layer][k]._activation;
                y_j = nn._avals[current_layer][j]._activation;

                sigderiv = y_j * (1.0f - y_j);

                if (i == nn._num_layers - 1)
                {
                    perr = -1.0f * (expected[j] - y_j);
                }
                else
                {
                    perr = 0.0f;

                    for (int n = 0; n < nn._top[next_layer]; n++)
                    {
                        perr += nn._avals[next_layer][n]._errterm * nn._wmatrix[current_layer]._weights[n][j];
                    }

                    perr *= -1.0f;
                }

                errterm = -1.0f * sigderiv * perr;

                nn._avals[current_layer][j]._errterm = errterm;

                weight_gradient = errterm * learningRate * y_k;

                nn._wmatrix[current_layer-1]._weights[j][k] += weight_gradient;
            }
        }
    }
}

void write_weight_images(struct Network nn)
{
    // monochrome
    for (int i = 0; i < nn._num_layers - 1; i++)
    {
        write_weight_image(nn, i);
    }
}

void write_weight_image(struct Network nn, int i)
{
    static int written = 0;
    unsigned char * img = (unsigned char*)malloc(sizeof(unsigned char) * nn._wmatrix[i]._c * nn._wmatrix[i]._r);

    for (int r = 0; r < nn._wmatrix[i]._r; r++)
    {
        for (int c = 0; c < nn._wmatrix[i]._c; c++)
        {
            img[r * nn._wmatrix[i]._c + c] = (int)(nn._wmatrix[i]._weights[r][c] * 255.0f);
        }
    }

    char file[20];
    sprintf(file, "imgs/data%d.png", written);

    int err = stbi_write_png(file, nn._wmatrix[i]._c, nn._wmatrix[i]._r, 1, img, 0);

    free(img);

    written++;
}