#include "Matrix.h"
#include "MNIST_DS.h"
#pragma once


class Network{
	int num_layers;
	int* sizes;
	Matrix* biases;
	Matrix* weights;
	Matrix* nabla_b;
	Matrix* nabla_w;

public:
	Network(int* sizes);
	~Network();

	Matrix feedforward(Matrix a);
	void SGD(MNIST_DS training_data, int epochs, int mini_batch_size, double eta, MNIST_DS test_data);
	void updateMiniBatch(MNIST_DS training_data, double eta);
	void backpropagation(vector<Matrix> x, vector<Matrix> y);
	void printBiases();
	void printWeights();
	Matrix sigmoid(Matrix& mtx);
	Matrix sigmoid_prime(Matrix& mtx);
	Matrix cost_derivative(vector<Matrix> output_activations, vector<Matrix> y);
};

