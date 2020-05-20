#include <iostream>
#include "Network.h"

using std::cin;
using std::cout;
using std::endl;

//Constructor
Network::Network(int* sizes) 
	:num_layers{ 3 }, sizes{ nullptr }, biases{ nullptr }, weights{ nullptr } {

	this->sizes = new int[num_layers];
	biases = new Matrix[num_layers - 1];
	weights = new Matrix[num_layers - 1];

	if (NULL != sizes && NULL != biases && NULL != weights) {

		//Initialize sizes
		for (size_t i{ 0 }; i < num_layers; i++)
			this->sizes[i] = sizes[i];

		//Extract sizes of matrices
		Matrix matrix_sizes{ 2, num_layers - 1 };
		for (size_t i{ 0 }; i < num_layers - 1; i++) {
			for (size_t j{ 0 }; j < num_layers - 1; j++) {
				//Get sizes from [:-1]
				if (i == 0)
					matrix_sizes[i * (num_layers - 1) + j] = sizes[j];
				else
					matrix_sizes[i * (num_layers - 1) + j] = sizes[j + 1];
			}
		}

		//Initialize weights
		Matrix weights_1{ static_cast<int>(matrix_sizes[0]), static_cast<int>(matrix_sizes[1]) };
		Matrix weights_2{ static_cast<int>(matrix_sizes[2]), static_cast<int>(matrix_sizes[3]) };
		weights[0] = weights_1;
		weights[1] = weights_2;

		//Initialize biases
		Matrix biases_1{ static_cast<int>(matrix_sizes[2]), 1 };
		Matrix biases_2{ static_cast<int>(matrix_sizes[3]), 1 };
		biases[0] = biases_1;
		biases[1] = biases_2;
	}
}

//Destructor
Network::~Network() {
	delete[] weights;
	delete[] biases;
	delete[] sizes;
}

//Print biases
void Network::printBiases() {
	for (size_t i{ 0 }; i < num_layers - 1; i++)
		cout << biases[i] << endl;
	cout << endl;
}

//Print weights
void Network::printWeights() {
	for (size_t i{ 0 }; i < num_layers - 1; i++)
		cout << weights[i] << endl;
	cout << endl;
}