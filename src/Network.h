#include "Matrix.h"
#include "MNIST.h"
#pragma once


class Network{
	int num_layers;		//Layers (3)
	int performance;	//Correctly classified
	Matrix* biases;		//All biases
	Matrix* weights;	//All weights
	Matrix* nabla_b;	//Bias gradient
	Matrix* nabla_w;	//weight gradient
	Matrix* delta_nabla_b;	//BP bias delta
	Matrix* delta_nabla_w;	//BP weight delta
	MNIST training_data;	//MNIST training data
	MNIST test;				//MNIST test data

public:
	Network(int* sizes);
	~Network();

	Matrix feedforward(const Matrix& a);
	void SGD(int epochs, int mini_batch_size, float eta);
	void updateMiniBatch(int mini_batch_index, float eta);
	void backpropagation(const Matrix& x, const Matrix& y);
	void printBiases();
	void printWeights();
	void evaluate();
	void classify();
	int argmax(Matrix& mtx) const;
	void loadWeightsBiases();
	Matrix sigmoid( Matrix& mtx);
	Matrix sigmoid_prime( Matrix& mtx);
	Matrix cost_derivative(Matrix output_activations, Matrix y);
};

