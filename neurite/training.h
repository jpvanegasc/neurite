#ifndef TRAINING_H
#define TRAINING_H

#include <math.h>
#include <stdio.h>

#include "neural_network.h"

void train();
int predict();

float sigmoid(float x);

// a1_i = s(W_ij * a0_j + b_i)
void forward_propagation_step(struct Layer *a1, struct Layer *a0);
void forward_propagation(struct Network *network);

float cost_function(struct Layer *output_layer, float *expected_output);

#endif // TRAINING_H
