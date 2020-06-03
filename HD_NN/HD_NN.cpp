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
    int* sizes = new int[n] {2, 3, 1};

    Network NN{ sizes };   
    //NN.printBiases();
    //NN.printWeights();
    Matrix input{ 1, 2 };
    //cout << NN.feedforward(input) << endl;
    NN.SGD(30, 10, 3.0f);


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

    //MNIST_DS test{ true };

    //for (size_t i{0 }; i < 10; i++) {
    //    cout << test.getImage(i+10) << endl;
    //    cout << test.getLabel(i+10) << endl;
    //}
    //cout << "\n--------------------------------------------------------------------" << endl;
    //test.mini_batches(10);
    //for (size_t i{ 0 }; i < 10; i++) {
    //    cout << test.getMiniBatchImages(1).at(i) << endl;
    //    cout << test.getMiniBatchLabels(1).at(i) << endl;
    //}
        

    return 0;
}
