#include <stdio.h>

#include "neural_network.h"

int main() {
    train();
    int result = predict();
    printf("Prediction: %d\n", result);
    return 0;
}
