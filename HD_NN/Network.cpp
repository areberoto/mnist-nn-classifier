#include <iostream>
#include <cmath>
#include "Network.h"
#include "MNIST_DS.h"

using std::cin;
using std::cout;
using std::endl;

//Constructor
Network::Network(int* sizes) 
	:num_layers{ 3 }, sizes{ nullptr }, biases{ nullptr }, weights{ nullptr }, nabla_b{ nullptr }, nabla_w{ nullptr } {

	this->sizes = new int[num_layers];
	biases = new Matrix[num_layers - 1];
	weights = new Matrix[num_layers - 1];
	nabla_b = new Matrix[num_layers - 1];
	nabla_w = new Matrix[num_layers - 1];

	if (NULL != sizes && NULL != biases && NULL != weights && NULL != nabla_b && NULL != nabla_w) {

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
				//Get sizes from [1:]
				else
					matrix_sizes[i * (num_layers - 1) + j] = sizes[j + 1];
			}
		}

		//Initialize weights
		Matrix weights_1{ static_cast<int>(matrix_sizes[1]), static_cast<int>(matrix_sizes[0]) };
		Matrix weights_2{ static_cast<int>(matrix_sizes[3]), static_cast<int>(matrix_sizes[2]) };
		weights[0] = weights_1;
		weights[1] = weights_2;

		//Create zero matrices for weights used in minibatch
		weights_1.zeros();
		weights_2.zeros();
		nabla_w[0] = weights_1;
		nabla_w[1] = weights_2;

		//Initialize biases
		Matrix biases_1{ 1, static_cast<int>(matrix_sizes[2]) };
		Matrix biases_2{ 1, static_cast<int>(matrix_sizes[3]) };
		biases[0] = biases_1;
		biases[1] = biases_2;

		//Create zero matrices for biases used in minibatch
		biases_1.zeros();
		biases_2.zeros();
		nabla_b[0] = biases_1;
		nabla_b[1] = biases_2;
	}
}

//Destructor
Network::~Network() {
	delete[] weights;
	delete[] biases;
	delete[] sizes;
}

//Feedforward
Matrix Network::feedforward(Matrix a) {
	for (size_t i{ 0 }; i < num_layers - 1; i++) {
		a = weights[i].dot(a) + biases[i];
		a.sigmoid();
	}	
	return a;
}

//SGD algorithm
void Network::SGD(MNIST_DS training_data, int epochs, int mini_batch_size, double eta, MNIST_DS test_data) {
	for (size_t i{ 0 }; i < epochs; i++) {
		training_data.randomSet(mini_batch_size);
		updateMiniBatch(training_data.getMiniBatchImages(), training_data.getMiniBatchLabels(), eta);
		cout << "Epoch [" << i + 1 << "] complete." << endl;
	}
}

//Update mini batch
void Network::updateMiniBatch(Matrix* images, unsigned char* labels, double eta) {

}

//Print biases
void Network::printBiases() {
	for (size_t i{ 0 }; i < num_layers - 1; i++) {
		cout << "Biases " << i + 1 << ": " << endl;
		cout << biases[i] << endl;
	}
	cout << endl;
}

//Print weights
void Network::printWeights() {
	for (size_t i{ 0 }; i < num_layers - 1; i++) {
		cout << "Weights " << i + 1 << ": " << endl;
		cout << weights[i] << endl;
	}
	cout << endl;
}