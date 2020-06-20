/*  Handwritten Digits Neural Network implemented in C++
    Author: Alberto Alvarez & Octavio Torres
    Date:   26th June, 2020
*/

#include <iostream>
#include "Network.h"
#include "MNIST.h"
#include "Matrix.h"

using std::cout;
using std::cin;
using std::endl;

int main()
{
    cout << "\tSHALLOW NEURAL NETWORK FOR HANDWRITTEN DIGITS RECOGNITION" << endl;
    int option{ 0 };
    int sizes[] = { 784, 30, 10 };
    Network classifier{ sizes };

    cout << "\nMENU:" << endl;
    cout << "1 - Classify handwritten digits" << endl;
    cout << "2 - Train neural network" << endl;
    cout << "3 - Exit" << endl;

    cout << "Option: ";
    cin >> option;

    while (option != 3) {
        switch (option) {
            case 1: {
                for (size_t i{ 0 }; i < 10; i++)
                    classifier.classify();
                break;
            }
            case 2: {
                int epochs{ 0 }, mini_batch{ 0 };
                float eta{ 0.0 };
                //NN.loadWeightsBiases();
                cout << "\nNumber of epochs: ";
                cin >> epochs;
                cout << "Mini batch size: ";;
                cin >> mini_batch;
                cout << "Learning rate: ";
                cin >> eta;
                cout << "\nTraining..." << endl;
                //classifier.SGD(epochs, mini_batch, eta);
                break;
            }
            case 3:
                return 0;
                break;
        }

        cout << "\nMENU:" << endl;
        cout << "1 - Classify handwritten digits" << endl;
        cout << "2 - Train neural network" << endl;
        cout << "3 - Exit" << endl;

        cout << "Option: ";
        cin >> option;
    }    

    return 0;
}