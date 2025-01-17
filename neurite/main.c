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

    struct Network network;
    network.input_layer = &input_layer;
    network.hidden_layers_size = 2;
    network.hidden_layers = (struct Layer **)malloc(2 * sizeof(struct Layer *));
    network.hidden_layers[0] = &hidden_layer_1;
    network.hidden_layers[1] = &hidden_layer_2;
    network.output_layer = &output_layer;

    for (int i = 0; i < INPUT_SIZE; i++) {
        input_layer.neurons[i] = five[i];
    }

    forward_propagation(&network);

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
