#include "neural_network.h"

void train() { printf("Training...\n"); }

int predict() {
    printf("Predicting...\n");
    return 0;
}

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

void initialize_layer(struct Layer *layer, int layer_size, int output_size) {
    layer->neurons_size = layer_size;
    layer->output_size = output_size;
    layer->neurons = (float *)malloc(layer_size * sizeof(float));

    layer->biases = (float *)malloc(output_size * sizeof(float));
    layer->weights = (float **)malloc(output_size * sizeof(float *));
    for (int i = 0; i < output_size; i++) {
        layer->weights[i] = (float *)malloc(layer_size * sizeof(float));
    }

    if (!layer->neurons || !layer->biases || !layer->weights) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
}

void initialize_network(struct Network *network, int input_size, int output_size,
                        int hidden_layers_size, int *hidden_layers_sizes) {

    network->hidden_layers_size = hidden_layers_size;
    network->input_layer = (struct Layer *)malloc(sizeof(struct Layer));
    network->output_layer = (struct Layer *)malloc(sizeof(struct Layer));
    network->hidden_layers =
        (struct Layer **)malloc(hidden_layers_size * sizeof(struct Layer *));

    initialize_layer(network->input_layer, input_size, hidden_layers_sizes[0]);
    initialize_layer(network->output_layer, output_size, 0);

    for (int i = 0; i < hidden_layers_size; i++) {
        network->hidden_layers[i] = (struct Layer *)malloc(sizeof(struct Layer));
        int next_layer_size =
            i == hidden_layers_size - 1 ? output_size : hidden_layers_sizes[i + 1];
        initialize_layer(network->hidden_layers[i], hidden_layers_sizes[i],
                         next_layer_size);
    }
}

void free_layer(struct Layer *layer) {
    free(layer->neurons);
    free(layer->biases);
    for (int i = 0; i < layer->output_size; i++) {
        free(layer->weights[i]);
    }
    free(layer->weights);
    free(layer);
}

void free_network(struct Network *network) {
    free_layer(network->input_layer);
    free_layer(network->output_layer);

    for (int i = 0; i < network->hidden_layers_size; i++) {
        free_layer(network->hidden_layers[i]);
    }
    free(network->hidden_layers);
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
