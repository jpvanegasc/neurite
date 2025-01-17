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

void train();

int predict();

float sigmoid(float x);

void initialize_layer(struct Layer *layer, int layer_size, int output_size);

void free_layer(struct Layer *layer, int output_size);

// a1_i = s(W_ij * a0_j + b_i)
void forward_propagation_step(struct Layer *a1, struct Layer *a0);
#endif // __NEURAL_NETWORK_H
