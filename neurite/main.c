#include <stdio.h>

#include "neural_network.h"
#include "temporary_values.h"

const int INPUT_SIZE = 784;
const int HIDDEN_1_SIZE = 16;
const int HIDDEN_2_SIZE = 16;
const int OUTPUT_SIZE = 10;

int main() {
    struct Network network;
    initialize_network(&network, INPUT_SIZE, OUTPUT_SIZE, 2,
                       (int[]){HIDDEN_1_SIZE, HIDDEN_2_SIZE});

    for (int i = 0; i < INPUT_SIZE; i++) {
        network.input_layer->neurons[i] = five[i];
    }

    forward_propagation(&network);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%f ", network.output_layer->neurons[i]);
    }

    free_network(&network);

    return 0;
}
