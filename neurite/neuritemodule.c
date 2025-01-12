#include <Python.h>

#include "neural_network.h"

static PyObject *train_method(PyObject *self, PyObject *args) {
    train();
    Py_RETURN_NONE;
}

static PyObject *predict_method(PyObject *self, PyObject *args) {
    int result = predict();
    return PyLong_FromLong(result);
}

static PyMethodDef neurite_methods[] = {
    {"train", train_method, METH_NOARGS, "Train the neural network"},
    {"predict", predict_method, METH_NOARGS,
     "Predict using the neural network"},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef c_neurite_module = {
    PyModuleDef_HEAD_INIT, "c_neurite", "Neural network C module", -1,
    neurite_methods};

PyMODINIT_FUNC PyInit_c_neurite(void) {
    return PyModule_Create(&c_neurite_module);
}
