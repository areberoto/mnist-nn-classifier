/*  Handwritten Digits Neural Network implemented in C++
    Author: Alberto Alvarez & Octavio Torres
    Date:   20th May, 2020
*/

#include <iostream>
#include "Network.h"
#include "Matrix.h"

using std::cout;
using std::cin;
using std::endl;

int main()
{
    const int n = 3;
    int* sizes = new int[n] {2, 3, 1};
    Matrix a{ 1, 2 };

    Network NN{ sizes };

    cout << "Input: " << endl;
    cout << a << endl;
    cout << "Print biases:" << endl;
    NN.printBiases();
    cout << "Print weights:" << endl;
    NN.printWeights();

    cout << "Dot: " << NN.feedforward(a) << endl;

    return 0;
}