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
    cout << "\tSHALLOW NEURAL NETWORK FOR HANDWRITTEN DIGITS RECOGNITION" << endl;
    int* sizes = new int[3] {784, 30, 10};

    Network NN{ sizes };   
    cout << "\nTraining..." << endl;
    NN.SGD(30, 100, 3.0f);

    return 0;
}
