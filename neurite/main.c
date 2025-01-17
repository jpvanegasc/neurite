#include <stdio.h>

#include "neural_network.h"
#include "temporary_values.h"

const int INPUT_SIZE = 784;
const int HIDDEN_1_SIZE = 16;
const int HIDDEN_2_SIZE = 16;
const int OUTPUT_SIZE = 10;

int main() {

    struct Layer input_layer, hidden_layer_1, hidden_layer_2, output_layer;

    initialize_layer(&input_layer, INPUT_SIZE, HIDDEN_1_SIZE);
    initialize_layer(&hidden_layer_1, HIDDEN_1_SIZE, HIDDEN_2_SIZE);
    initialize_layer(&hidden_layer_2, HIDDEN_2_SIZE, OUTPUT_SIZE);
    initialize_layer(&output_layer, OUTPUT_SIZE, 0);

    for (int i = 0; i < INPUT_SIZE; i++) {
        input_layer.neurons[i] = five[i];
    }

    printf("Input layer:\n");
    forward_propagation_step(&hidden_layer_1, &input_layer);
    printf("Hidden layer 1:\n");
    forward_propagation_step(&hidden_layer_2, &hidden_layer_1);
    printf("Hidden layer 2:\n");
    forward_propagation_step(&output_layer, &hidden_layer_2);

    // Print output_layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%f ", output_layer.neurons[i]);
    }

    free_layer(&input_layer, HIDDEN_1_SIZE);
    free_layer(&hidden_layer_1, HIDDEN_2_SIZE);
    free_layer(&hidden_layer_2, OUTPUT_SIZE);
    free_layer(&output_layer, 0);

    return 0;
}
