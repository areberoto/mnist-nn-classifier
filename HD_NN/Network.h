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
	Matrix* delta_nabla_b;
	Matrix* delta_nabla_w;
	MNIST_DS training_data;

public:
	Network(int* sizes);
	~Network();

	Matrix feedforward(Matrix a);
	void SGD(int epochs, int mini_batch_size, double eta);
	void updateMiniBatch(int mini_batch_index, double eta);
	void backpropagation(Matrix x, Matrix y);
	void printBiases();
	void printWeights();
	Matrix sigmoid(Matrix& mtx);
	Matrix sigmoid_prime(Matrix& mtx);
	vector<Matrix> sigmoid(vector<Matrix>& mtx);
	vector<Matrix> sigmoid_prime(vector<Matrix>& mtx);
	Matrix cost_derivative(Matrix output_activations, Matrix y);
};

