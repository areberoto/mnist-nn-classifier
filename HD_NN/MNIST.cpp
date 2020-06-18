#include "MNIST.h"
#include "Matrix.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <random>
#include <algorithm>

using std::endl;
using std::cout;
using std::string;
using std::vector;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;

//Constructor
MNIST::MNIST(string data)
    : file{ data }, n{ 0 }, m{ 0 }, magic_number{ 0 }, number_of_items{ 0 }, mini_batch_size{ 0 },
    image_dat_set{ nullptr }, label_dat_set{ nullptr }, mini_batch_imag{ nullptr }, mini_batch_label{ nullptr }{

    string nameImagesFile{};
    string nameLabelsFile{};

    if (file == "train") {
        nameImagesFile = "train-images.idx3-ubyte";
        nameLabelsFile = "train-labels.idx1-ubyte";
    }
    else {
        nameImagesFile = "t10k-images.idx3-ubyte";
        nameLabelsFile = "t10k-labels.idx1-ubyte";
    }

    //Read images
    ifstream file_imag{ nameImagesFile, ios::in | ios::binary | ios::ate };

    if (!file_imag) {
        cerr << "Error trying to open the file!\n" << endl;
        exit(1);
    }

    ifstream::pos_type size = file_imag.tellg();
    file_imag.seekg(0, ios::beg);

    file_imag.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
    magic_number = reverseInt(magic_number);
    file_imag.read(reinterpret_cast<char*>(&number_of_items), sizeof(int));
    number_of_items = reverseInt(number_of_items);
    file_imag.read(reinterpret_cast<char*>(&n), sizeof(int));
    n = reverseInt(n);
    file_imag.read(reinterpret_cast<char*>(&m), sizeof(int));
    m = reverseInt(m);

    unsigned char pixel{ 0 };
    Matrix image{ n * m, 1 };
    image_dat_set = new Matrix[number_of_items];

    if (image_dat_set != NULL) {
        cout << "\nLoading data set of " << ((file == "train") ? "training" : "test") << " images... ";
        for (size_t i{ 0 }; i < number_of_items; i++) {
            for (size_t j{ 0 }; j < n * m; j++) {
                file_imag.read(reinterpret_cast<char*>(&pixel), sizeof(unsigned char));
                image[j] = static_cast<float>(pixel);
            }
            image_dat_set[i] = image;
        }
        cout << "DONE!" << endl;
        file_imag.close();
    }
    else {
        cerr << "Error allocating dynamic memory!\n" << endl;
        file_imag.close();
        exit(1);
    }

    //Read labels
    ifstream file_labels{ nameLabelsFile, ios::in | ios::binary | ios::ate };

    if (!file_labels) {
        cerr << "Error trying to open the file!\n" << endl;
        exit(1);
    }

    size = file_labels.tellg();
    file_labels.seekg(0, ios::beg);

    file_labels.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
    magic_number = reverseInt(magic_number);
    file_labels.read(reinterpret_cast<char*>(&number_of_items), sizeof(int));
    number_of_items = reverseInt(number_of_items);

    unsigned char label{ 0 };
    Matrix temp{ 10, 1 };
    label_dat_set = new Matrix[number_of_items];

    if (label_dat_set != NULL) {
        cout << "Loading data set of " << ((file == "train") ? "training" : "test") << " labels... ";
        for (size_t i{ 0 }; i < number_of_items; i++) {
            file_labels.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));

            for (size_t j{ 0 }; j < 10; j++)
                temp[j] = (j == static_cast<int>(label)) ? 1.0f : 0.0f;
            label_dat_set[i] = temp;
        }

        cout << "DONE!" << endl;
        file_labels.close();
    }
    else {
        cerr << "Error allocating dynamic memory!\n" << endl;
        file_labels.close();
        exit(1);
    }
}

//Copy constructor
MNIST::MNIST(const MNIST& data)
    : file{ data.file }, n{ data.n }, m{ data.m }, magic_number{ data.magic_number },
    number_of_items{ data.number_of_items }, mini_batch_size{ data.mini_batch_size }{

    image_dat_set = new Matrix[number_of_items];
    label_dat_set = new Matrix[number_of_items];
    mini_batch_imag = new Matrix[number_of_items];
    mini_batch_label = new Matrix[number_of_items];

    if (image_dat_set != NULL && label_dat_set != NULL && mini_batch_imag != NULL && mini_batch_label != NULL) {
        memcpy(image_dat_set, data.image_dat_set, number_of_items * sizeof(Matrix));
        memcpy(label_dat_set, data.label_dat_set, number_of_items * sizeof(Matrix));
        memcpy(mini_batch_imag, data.mini_batch_imag, number_of_items * sizeof(Matrix));
        memcpy(mini_batch_label, data.mini_batch_label, number_of_items * sizeof(Matrix));
    }
    else {
        cerr << "Error allocating dynamic memory!\n" << endl;
        exit(1);
    }
}

MNIST::~MNIST() {
    if (number_of_items != 0) {
        delete[] image_dat_set;
        delete[] label_dat_set;
        delete[] mini_batch_imag;
        delete[] mini_batch_label;
    }
}

//Convert raw bytes to int
int MNIST::reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

//Get number of items
int MNIST::get_number_items() {
    return number_of_items;
}

//Get image
Matrix MNIST::getImage(int index) {
    return image_dat_set[index];
}

//Get label
Matrix MNIST::getLabel(int index) {
    return label_dat_set[index];
}

void MNIST::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());
    vector<int> indexes;
    indexes.reserve(number_of_items);

    for (size_t i{ 0 }; i < number_of_items; i++)
        indexes.push_back(i);
    std::shuffle(indexes.begin(), indexes.end(), g);

    Matrix* temp_image = new Matrix[number_of_items];
    Matrix* temp_label = new Matrix[number_of_items];

    if (temp_image != NULL && temp_label != NULL) {
        for (size_t i{ 0 }; i < number_of_items; i++) {
            temp_image[i] = image_dat_set[indexes.at(i)];
            temp_label[i] = label_dat_set[indexes.at(i)];
        }

        for (size_t i{ 0 }; i < number_of_items; i++) {
            image_dat_set[i] = temp_image[i];
            label_dat_set[i] = temp_label[i];
        }
        //memcpy(image_dat_set, temp_image, number_of_items * sizeof(Matrix));
        //memcpy(label_dat_set, temp_label, number_of_items * sizeof(Matrix));

        delete[] temp_image;
        delete[] temp_label;
    }
    else {
        cerr << "Error allocating dynamic memory!\n" << endl;
        exit(1);
    }
}

void MNIST::mini_batches(int n) {
    mini_batch_size = n;
    int index{ 0 };

    if (mini_batch_imag != NULL && mini_batch_label != NULL) {
        delete[] mini_batch_imag;
        delete[] mini_batch_label;
    }

    mini_batch_imag = new Matrix[number_of_items];
    mini_batch_label = new Matrix[number_of_items];

    if (mini_batch_imag != NULL && mini_batch_label != NULL) {
        for (size_t i{ 0 }; i < number_of_items / mini_batch_size; i++) {
            for (int j{ index }; j < index + mini_batch_size; j++) {
                mini_batch_imag[i * mini_batch_size + (j % mini_batch_size)] = image_dat_set[j];
                mini_batch_label[i * mini_batch_size + (j % mini_batch_size)] = label_dat_set[j];
            }
            index += mini_batch_size;
        }
    }
    else {
        cerr << "Error allocating dynamic memory!\n" << endl;
        exit(1);
    }    
}

int MNIST::get_mini_batch_size() {
    return mini_batch_size;
}

Matrix* MNIST::getMiniBatchImages(int index) {
    Matrix* temp = new Matrix[mini_batch_size];
    for (size_t i{ 0 }; i < mini_batch_size; i++)
        temp[i] = mini_batch_imag[index * mini_batch_size + i];
    return temp;
}

Matrix* MNIST::getMiniBatchLabels(int index) {
    Matrix* temp = new Matrix[mini_batch_size];
    for (size_t i{ 0 }; i < mini_batch_size; i++)
        temp[i] = mini_batch_label[index * mini_batch_size + i];
    return temp;
}