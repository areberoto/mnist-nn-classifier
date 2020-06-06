#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include "Network.h"
#include "MNIST_DS.h"

using std::cin;
using std::cout;
using std::endl;
using std::vector;

//Constructor
Network::Network(int* size_layers) 
	:num_layers{ 3 }, sizes{ nullptr }, biases{ nullptr }, weights{ nullptr }, nabla_b{ nullptr },
	nabla_w{ nullptr }, delta_nabla_b{ nullptr }, delta_nabla_w{ nullptr }, training_data{ true }, test{ false }{

	sizes = new int[num_layers];
	biases = new Matrix[2];
	weights = new Matrix[2];
	nabla_b = new Matrix[2];
	nabla_w = new Matrix[2];
	delta_nabla_b = new Matrix[2];
	delta_nabla_w = new Matrix[2];

	if (NULL != sizes && NULL != biases && NULL != weights && NULL != nabla_b && NULL != nabla_w
		&& NULL != delta_nabla_b && NULL != delta_nabla_w) {

		//Initialize sizes
		for (size_t i{ 0 }; i < num_layers; i++)
			sizes[i] = size_layers[i];

		//Extract sizes of matrices
		Matrix matrix_sizes{ 2, 2 };
		for (size_t i{ 0 }; i < 2; i++) {
			for (size_t j{ 0 }; j < 2; j++) {
				//Get sizes from [:-1]
				if (i == 0)
					matrix_sizes[i * 2 + j] = sizes[j];
				//Get sizes from [1:]
				else
					matrix_sizes[i * 2 + j] = sizes[j + 1];
			}
		}

		//Initialize weights
		Matrix weights_1{ static_cast<int>(matrix_sizes[1]), static_cast<int>(matrix_sizes[0]) };
		Matrix weights_2{ static_cast<int>(matrix_sizes[3]), static_cast<int>(matrix_sizes[2]) };
		weights[0] = weights_1;
		weights[1] = weights_2;

		//Create matrices for weights used in minibatch
		nabla_w[0] = weights_1;
		nabla_w[1] = weights_2;
		delta_nabla_w[0] = weights_1;
		delta_nabla_w[1] = weights_2;

		//Initialize biases
		Matrix biases_1{ static_cast<int>(matrix_sizes[2]), 1 };
		Matrix biases_2{ static_cast<int>(matrix_sizes[3]), 1 };
		biases[0] = biases_1;
		biases[1] = biases_2;

		//Create matrices for biases used in minibatch
		nabla_b[0] = biases_1;
		nabla_b[1] = biases_2;
		delta_nabla_b[0] = biases_1;
		delta_nabla_b[1] = biases_2;
	}
}

//Destructor
Network::~Network() {
	delete[] weights;
	delete[] biases;
	delete[] sizes;
	delete[] nabla_b;
	delete[] nabla_w;
	delete[] delta_nabla_b;
	delete[] delta_nabla_w;
}

//Feedforward
Matrix Network::feedforward(Matrix a) {
	for (size_t i{ 0 }; i < 2; i++) {
		a = weights[i] * a + biases[i];
		a = sigmoid(a);
	}	
	return a;
}

//SGD algorithm
void Network::SGD( int epochs, int mini_batch_size, float eta) {
	for (size_t i{ 0 }; i < epochs; i++) {
		training_data.shuffle();
		training_data.mini_batches(mini_batch_size);

		for (size_t j{ 0 }; j < 1000; j++)
		//for (size_t j{ 0 }; j < training_data.get_number_items() / mini_batch_size; j++) {
			updateMiniBatch(j, eta);

		cout << "Epoch [" << i + 1 << "] complete." << endl;
		evaluate();
	}
}

void Network::evaluate() {
	Matrix result;
	int index{ 0 }, y{ 0 };
	int count{ 0 };
	for (size_t i{ 0 }; i < test.get_number_items(); i++) {
		result = feedforward(test.getImage(i));
		index = argmax(result);
		result = test.getLabel(i);
		y = argmax(result);
		if (index == y)
			count++;
	}
	cout << count << "/" << test.get_number_items() << endl;
}

int Network::argmax(Matrix& mtx) const{
	double max{ mtx[0] };
	int index{ 0 };
	for (size_t i{ 1 }; i < mtx.getSize(); i++) {
		if (max < mtx[i]) {
			max = mtx[i];
			index = i;
		}			
	}
	return index;
}

//Update mini batch
void Network::updateMiniBatch(int mini_batch_index, float eta) {
	nabla_b[0].zeros();
	nabla_b[1].zeros();
	nabla_w[0].zeros();
	nabla_w[1].zeros();
	
	for (size_t i{ 0 }; i < training_data.get_mini_batch_size(); i++) {
		backpropagation(training_data.getMiniBatchImages(mini_batch_index).at(i), training_data.getMiniBatchLabels(mini_batch_index).at(i));
		for (size_t j{ 0 }; j < num_layers - 1; j++) {
			nabla_b[j] = nabla_b[j] + delta_nabla_b[j];
			nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
		}
	}
		
	for (size_t i{ 0 }; i < num_layers - 1; i++) {
		weights[i] = weights[i] - (nabla_w[i] * (eta / training_data.get_mini_batch_size()));
		biases[i] = biases[i] - (nabla_b[i] * (eta / training_data.get_mini_batch_size()));
	}
}

//Backpropagation
void Network::backpropagation(const Matrix& x, const Matrix& y) {
	
	//feedforward
	Matrix activation{ x };
	vector<Matrix> activations;
	activations.push_back(x); //list to store all the activations, layer by layer
	vector<Matrix> zs; //list to store all the z vectors, layer by layer
	Matrix z;

	for (size_t i{ 0 }; i < num_layers - 1; i++) {
		z = (weights[i] * activation) + biases[i];
		zs.push_back(z);
		activation = sigmoid(z);
		activations.push_back(activation);
	}

	//Backward pass
	Matrix delta = cost_derivative(activations.at(2), y) ^ sigmoid_prime(zs.at(1));

	delta_nabla_b[1] = delta;
	delta_nabla_w[1] = delta * ~activations.at(1);

	z = zs.at(0);
	Matrix sp{ sigmoid_prime(z) };
	delta = (~weights[1] * delta) ^ sp;
	delta_nabla_b[0] = delta;
	delta_nabla_w[0] = delta * ~activations.at(0);
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
	Matrix temp{ mtx.getRows(), mtx.getColumns() };
	for (size_t i{ 0 }; i < mtx.getSize(); i++)
		temp[i] = 1.0f / (1.0f + exp(-1.0f*mtx[i]));
	return temp;
}

//Derivative of the sigmoid function
Matrix Network::sigmoid_prime( Matrix& mtx) {
	Matrix temp{ sigmoid(mtx) };
	for (size_t i{ 0 }; i < mtx.getSize(); i++)
		temp[i] = temp[i] * (1.0f - temp[i]);
	return temp;
}

//Cost derivative
Matrix Network::cost_derivative(Matrix output_activations, Matrix y) {
	Matrix temp{output_activations.getRows(), output_activations.getColumns()};
	for (size_t i{ 0 }; i < output_activations.getSize(); i++)
		temp[i] = output_activations[i] - y[i];
	return temp;
}