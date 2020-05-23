#include "Matrix.h"
#include "MNIST_DS.h"
#pragma once


class Network{
	int num_layers;
	int* sizes;
	Matrix* biases;
	Matrix* weights;

public:
	Network(int* sizes);
	~Network();

	Matrix feedforward(Matrix a);
	void SGD(MNIST_DS training_data, int epochs, int mini_batch_size, double eta, MNIST_DS test_data);
	void updateMiniBatch(Matrix* images, unsigned char* labels, double eta);
	void printBiases();
	void printWeights();
};

