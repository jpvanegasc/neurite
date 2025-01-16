#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "temporary_values.h"

const int INPUT_SIZE = 784;
const int HIDDEN_1_SIZE = 16;
const int HIDDEN_2_SIZE = 16;
const int OUTPUT_SIZE = 10;

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

int main() {
    float input_layer[INPUT_SIZE] = {0};                      // 28x28 grid
    float input_to_hidden_1[INPUT_SIZE][HIDDEN_1_SIZE] = {0}; // 16 neurons
    float hidden_layer_1[16] = {0};                           // 16 neurons
    float hidden_1_to_hidden_2[16][16] = {0};                 // 16 neurons
    float hidden_layer_2[16] = {0};                           // 16 neurons
    float hidden_2_to_output[16][OUTPUT_SIZE] = {0}; // 10 possible digits
    float output_layer[OUTPUT_SIZE] = {0};           // 10 possible digits

    // Load input_layer with pixel values
    for (int i = 0; i < INPUT_SIZE; i++) {
        input_layer[i] = five[i];
    }

    // Calculate hidden_layer_1
    for (int i = 0; i < HIDDEN_1_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_layer_1[i] += input_layer[j] * input_to_hidden_1[j][i];
        }
        hidden_layer_1[i] = sigmoid(hidden_layer_1[i]);
    }

    // Calculate hidden_layer_2
    for (int i = 0; i < HIDDEN_2_SIZE; i++) {
        for (int j = 0; j < HIDDEN_1_SIZE; j++) {
            hidden_layer_2[i] += hidden_layer_1[j] * hidden_1_to_hidden_2[j][i];
        }
        hidden_layer_2[i] = sigmoid(hidden_layer_2[i]);
    }

    // Calculate output_layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_2_SIZE; j++) {
            output_layer[i] += hidden_layer_2[j] * hidden_2_to_output[j][i];
        }
        output_layer[i] = sigmoid(output_layer[i]);
    }

    // Print output_layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%f ", output_layer[i]);
    }

    return 0;
}
