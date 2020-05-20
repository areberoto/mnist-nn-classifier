#include "Matrix.h"
#pragma once


class Network{
	int num_layers;
	int* sizes;
	Matrix* biases;
	Matrix* weights;

public:
	Network(int* sizes);
	~Network();

	void printBiases();
	void printWeights();
};

