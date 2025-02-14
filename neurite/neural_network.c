#include "neural_network.h"

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
