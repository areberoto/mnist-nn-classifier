#include <iostream>
#include <cmath>
#include <vector>
#include "Network.h"
#include "MNIST_DS.h"

using std::cin;
using std::cout;
using std::endl;
using std::vector;

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
		a = sigmoid(a);
	}	
	return a;
}

//SGD algorithm
void Network::SGD(MNIST_DS training_data, int epochs, int mini_batch_size, double eta, MNIST_DS test_data) {

	for (size_t i{ 0 }; i < epochs; i++) {
		training_data.shuffle();
		training_data.mini_batches(mini_batch_size);

		/*for (size_t j{ 0 }; j < 10; j++) {
			cout << training_data.getMiniBatchImages(0).at(j) << endl;
			cout << training_data.getMiniBatchLabels(0).at(j) << endl;
		}*/

		updateMiniBatch(training_data, eta);
		cout << "Epoch [" << i + 1 << "] complete." << endl;
	}
}

//Update mini batch
void Network::updateMiniBatch(MNIST_DS training_data, double eta) {

	for (size_t i{ 0 }; i < training_data.get_mini_batch_size(); i++) {
		//backpropagation(training_data.getMiniBatchImages(i), training_data.getMiniBatchLabels(i));
	}
}

//Backpropagation
void Network::backpropagation(vector<Matrix> x, vector<Matrix> y) {
	Matrix nabla_b_0{ nabla_b[0] };
	Matrix nabla_b_1{ nabla_b[1] };
	nabla_b_0.zeros();
	nabla_b_1.zeros();

	Matrix nabla_w_0{ nabla_w[0] };
	Matrix nabla_w_1{ nabla_w[1] };
	nabla_w_0.zeros();
	nabla_w_1.zeros();
	
	//feedforward
	vector<Matrix> activation = x;
	vector<vector<Matrix>> activations;
	activations.push_back(x); //list to store all the activations, layer by layer
	vector<vector<Matrix>> zs; //list to store all the z vectors, layer by layer
	vector<Matrix> z;

	for (size_t i{ 0 }; i < num_layers - 1; i++) {
		for (size_t j{ 0 }; j < x.size(); j++)
			z.push_back(weights[i].dot(activation.at(j)) + biases[i]);
		zs.push_back(z);
		
		for (size_t j{ 0 }; j < x.size(); j++) {
			activation.at(j) = sigmoid(z.at(j));
		}
		activations.push_back(activation);
	}

	//Backward pass
	cost_derivative(activations.back(), y);


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

//Sigmoid function
Matrix Network::sigmoid( Matrix& mtx) {
	Matrix temp{ 1, mtx.getSize() };
	for (size_t i{ 0 }; i < mtx.getSize(); i++)
		temp[i] = 1.0 / (1.0 + exp(mtx[i] * -1));
	return temp;
}

//Derivative of the sigmoid function
Matrix Network::sigmoid_prime(Matrix& mtx) {
	Matrix temp{ sigmoid(mtx) };
	for (size_t i{ 0 }; i < mtx.getSize(); i++)
		temp[i] = temp[i] * (1 - temp[i]);
	return temp;
}

//Cost derivative
Matrix Network::cost_derivative(vector<Matrix> output_activations, vector<Matrix> y) {

	Matrix temp{ 1, 10 };
	for (size_t i{ 0 }; i < output_activations.size(); i++)
		//temp[i] = output_activations[i] - y[i];
	return temp;
}