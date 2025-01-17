#ifndef __NEURAL_NETWORK_H
#define __NEURAL_NETWORK_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct Layer {
    int neurons_size;
    float *neurons;
    float *biases;
    float **weights;
};

struct Network {
    struct Layer *input_layer, *output_layer;
    int hidden_layers_size;
    struct Layer **hidden_layers;
};

void train();

int predict();

float sigmoid(float x);

void initialize_layer(struct Layer *layer, int layer_size, int output_size);

void free_layer(struct Layer *layer, int output_size);

// a1_i = s(W_ij * a0_j + b_i)
void forward_propagation_step(struct Layer *a1, struct Layer *a0);

void forward_propagation(struct Network *network);
#endif // __NEURAL_NETWORK_H
