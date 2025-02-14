#include "training.h"

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

void train() { printf("Training...\n"); }

int predict() {
    printf("Predicting...\n");
    return 0;
}

void forward_propagation_step(struct Layer *a1, struct Layer *a0) {
    for (int i = 0; i < a1->neurons_size; i++) {
        float sum = 0;
        for (int j = 0; j < a0->neurons_size; j++) {
            sum += a0->neurons[i] * a0->weights[i][j];
        }
        a1->neurons[i] = sigmoid(sum + a0->biases[i]);
    }
}

void forward_propagation(struct Network *network) {
    forward_propagation_step(network->hidden_layers[0], network->input_layer);

    for (int i = 1; i < network->hidden_layers_size; i++) {
        forward_propagation_step(network->hidden_layers[i],
                                 network->hidden_layers[i - 1]);
    }

    forward_propagation_step(network->output_layer,
                             network->hidden_layers[network->hidden_layers_size - 1]);
}

float cost_function(struct Layer *output_layer, float *expected_output) {
    float cost = 0;
    for (int i = 0; i < output_layer->neurons_size; i++) {
        cost += pow(output_layer->neurons[i] - expected_output[i], 2);
    }
    return cost;
}
