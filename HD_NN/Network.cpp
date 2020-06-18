#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include "Network.h"
#include "MNIST.h"

using std::cin;
using std::cout;
using std::endl;
using std::string;

//Constructor
Network::Network(int* size_layers) 
	:num_layers{ 3 }, performance{ 0 }, biases{ nullptr }, weights{ nullptr }, nabla_b{ nullptr },
	nabla_w{ nullptr }, delta_nabla_b{ nullptr }, delta_nabla_w{ nullptr }, training_data{ "train" }, test{ "test" }{

	biases = new Matrix[2];
	weights = new Matrix[2];
	nabla_b = new Matrix[2];
	nabla_w = new Matrix[2];
	delta_nabla_b = new Matrix[2];
	delta_nabla_w = new Matrix[2];

	if (NULL != biases && NULL != weights && NULL != nabla_b && NULL != nabla_w
		&& NULL != delta_nabla_b && NULL != delta_nabla_w) {

		//Initialize weights
		Matrix weights_1{ size_layers[1], size_layers[0] };
		Matrix weights_2{ size_layers[2], size_layers[1] };
		weights[0] = weights_1;
		weights[1] = weights_2;

		//Create matrices for weights used in minibatch
		nabla_w[0] = weights_1;
		nabla_w[1] = weights_2;
		delta_nabla_w[0] = weights_1;
		delta_nabla_w[1] = weights_2;

		//Initialize biases
		Matrix biases_1{ size_layers[1], 1 };
		Matrix biases_2{ size_layers[2], 1 };
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
	delete[] nabla_b;
	delete[] nabla_w;
	delete[] delta_nabla_b;
	delete[] delta_nabla_w;
}

//Feedforward
Matrix Network::feedforward(const Matrix& a) {
	Matrix output{a};
	for (size_t i{ 0 }; i < 2; i++) {
		output = weights[i] * output + biases[i];
		output = sigmoid(output);
	}	
	return output;
}

//SGD algorithm
void Network::SGD( int epochs, int mini_batch_size, float eta) {
	for (size_t i{ 0 }; i < epochs; i++) {
		training_data.shuffle();
		training_data.mini_batches(mini_batch_size);

		for (size_t j{ 0 }; j < 150; j++)
			updateMiniBatch(j, eta);

		cout << "Epoch [" << i + 1 << "] complete." << endl;
		evaluate();
	}
}

void Network::evaluate() {
	Matrix result{};
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

	if (count > performance) {
		performance = count;
		weights[0].saveMatrix("weights_0");
		weights[1].saveMatrix("weights_1");
		biases[0].saveMatrix("biases_0");
		biases[1].saveMatrix("biases_1");
	}
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
		Matrix* images;
		Matrix* labels;
		images = training_data.getMiniBatchImages(mini_batch_index);
		labels = training_data.getMiniBatchLabels(mini_batch_index);
		backpropagation(images[i], labels[i]);
		for (size_t j{ 0 }; j < 2; j++) {
			nabla_b[j] = nabla_b[j] + delta_nabla_b[j];
			nabla_w[j] = nabla_w[j] + delta_nabla_w[j];
		}
		delete[] images;
		delete[] labels;
	}
		
	for (size_t i{ 0 }; i < 2; i++) {
		weights[i] = weights[i] - (nabla_w[i] * (eta / training_data.get_mini_batch_size()));
		biases[i] = biases[i] - (nabla_b[i] * (eta / training_data.get_mini_batch_size()));
	}
}

//Backpropagation
void Network::backpropagation(const Matrix& x, const Matrix& y) {
	delta_nabla_b[0].zeros();
	delta_nabla_b[1].zeros();
	delta_nabla_w[0].zeros();
	delta_nabla_w[1].zeros();

	//feedforward
	Matrix activation{ x };
	Matrix* activations = new Matrix[num_layers];
	activations[0] = x; //list to store all the activations, layer by layer
	Matrix* zs = new Matrix[2]; //list to store all the z vectors, layer by layer
	Matrix z{};

	for (size_t i{ 0 }; i < 2; i++) {
		z = (weights[i] * activation) + biases[i];
		zs[i] = z;
		activation = sigmoid(z);
		activations[i + 1] = activation;
	}

	//Backward pass
	Matrix delta = cost_derivative(activations[2], y) ^ sigmoid_prime(zs[1]);

	delta_nabla_b[1] = delta;
	delta_nabla_w[1] = delta * ~activations[1];

	z = zs[0];
	Matrix sp{ sigmoid_prime(z) };
	delta = (~weights[1] * delta) ^ sp;
	delta_nabla_b[0] = delta;
	delta_nabla_w[0] = delta * ~activations[0];

	delete[] activations;
	delete[] zs;
}

//Print biases
void Network::printBiases() {
	for (size_t i{ 0 }; i < 2; i++) {
		cout << "Biases " << i + 1 << ": " << endl;
		cout << biases[i] << endl;
	}
	cout << endl;
}

//Print weights
void Network::printWeights() {
	for (size_t i{ 0 }; i < 2; i++) {
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

void Network::loadWeightsBiases() {
	weights[0].readMatrix("weights_0");
	weights[1].readMatrix("weights_1");
	biases[0].readMatrix("biases_0");
	biases[1].readMatrix("biases_1");
}