/*  Handwritten Digits Neural Network implemented in C++
    Author: Alberto Alvarez & Octavio Torres
    Date:   20th May, 2020
*/

#include <iostream>
#include "Network.h"
#include "MNIST_DS.h"
#include "Matrix.h"

using std::cout;
using std::cin;
using std::endl;

int main()
{
    const int n = 3;
    int* sizes = new int[n] {784, 30, 10};

    MNIST_DS training_data{ true };
    MNIST_DS test_data{ false };
    Network NN{ sizes };    
    NN.SGD(training_data, 30, 10, 3.0, test_data);

    //int* sizes = new int[n] {2, 3, 1};
    //Network NN{ sizes };
    //Matrix input{ inputImage.getImage(0) };

    //Matrix input{ 1, 2 };

    //cout << "BIASES:" << endl;
    //NN.printBiases();
    //cout << "WEIGHTS:" << endl;
    //NN.printWeights();   

    //cout << "Input: " << input << endl;
    //cout << "Output: " << NN.feedforward(input);
    return 0;
}
