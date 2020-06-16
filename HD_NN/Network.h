#include "Matrix.h"
#include "MNIST_DS.h"
#pragma once


class Network{
	int num_layers;
	Matrix* biases;
	Matrix* weights;
	Matrix* nabla_b;
	Matrix* nabla_w;
	Matrix* delta_nabla_b;
	Matrix* delta_nabla_w;
	MNIST_DS training_data;
	MNIST_DS test;

public:
	Network(int* sizes);
	~Network();

	Matrix feedforward(Matrix a);
	void SGD(int epochs, int mini_batch_size, float eta);
	void updateMiniBatch(int mini_batch_index, float eta);
	void backpropagation(const Matrix& x, const Matrix& y);
	void printBiases();
	void printWeights();
	void evaluate();
	int argmax(Matrix& mtx) const;
	Matrix sigmoid( Matrix& mtx);
	Matrix sigmoid_prime( Matrix& mtx);
	Matrix cost_derivative(Matrix output_activations, Matrix y);
};

