#include "neural_network.h"

void train() { printf("Training...\n"); }

int predict() {
    printf("Predicting...\n");
    return 0;
}

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

void initialize_layer(struct Layer *layer, int layer_size, int output_size) {
    layer->neurons_size = layer_size;
    layer->neurons = (float *)malloc(layer_size * sizeof(float));

    layer->biases = (float *)malloc(output_size * sizeof(float));
    layer->weights = (float **)malloc(output_size * sizeof(float *));
    for (int i = 0; i < output_size; i++) {
        layer->weights[i] = (float *)malloc(layer_size * sizeof(float));
    }
}

void free_layer(struct Layer *layer, int output_size) {
    free(layer->neurons);
    free(layer->biases);
    for (int i = 0; i < output_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
}

void forward_propagation_step(struct Layer *a1, struct Layer *a0) {
    for (int i = 0; i < a1->neurons_size; i++) {
        float sum = 0;
        for (int j = 0; j < a0->neurons_size; j++) {
            sum += a0->neurons[i] * a0->weights[i][j];
        }
        a1->neurons[i] = sigmoid(sum + a1->biases[i]);
    }
}

void forward_propagation(struct Network *network) {
    forward_propagation_step(network->hidden_layers[0], network->input_layer);

    for (int i = 1; i < network->hidden_layers_size; i++) {
        forward_propagation_step(network->hidden_layers[i],
                                 network->hidden_layers[i - 1]);
    }

    forward_propagation_step(
        network->output_layer,
        network->hidden_layers[network->hidden_layers_size - 1]);
}
