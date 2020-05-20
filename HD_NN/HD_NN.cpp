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
    
    Matrix input{ 1, 2 };
    Network NN{ sizes };

    cout << "Input: " << endl;
    cout << input << endl;
    cout << "BIASES:" << endl;
    NN.printBiases();
    cout << "WEIGHTS:" << endl;
    NN.printWeights();

    cout << "Dot: " << NN.feedforward(input) << endl;

    return 0;
}
