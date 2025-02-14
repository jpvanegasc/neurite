#ifndef __NEURAL_NETWORK_H
#define __NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>

struct Layer {
    int neurons_size;
    int output_size;
    float *neurons;
    float *biases;
    float **weights;
};

struct Network {
    struct Layer *input_layer, *output_layer;
    int hidden_layers_size;
    struct Layer **hidden_layers;
};

void initialize_layer(struct Layer *layer, int layer_size, int output_size);
void initialize_network(
    struct Network *network, int input_size, int output_size, int hidden_layers_size,
    int *hidden_layers_sizes
);

void free_layer(struct Layer *layer);
void free_network(struct Network *network);

#endif // __NEURAL_NETWORK_H
