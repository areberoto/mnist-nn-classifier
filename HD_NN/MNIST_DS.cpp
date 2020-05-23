#include <iostream>
#include <string>
#include <fstream>
#include "Matrix.h"
#include <ctime>
#include <cstdlib>
#include "MNIST_DS.h"

using std::endl;
using std::cout;
using std::string;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;

//Constructor
MNIST_DS::MNIST_DS(bool flag)
    : train{ flag }, n{ 0 }, m{ 0 }, magic_number{ 0 }, number_of_items{ 0 }, image_dat_set{ nullptr },
    label_dat_set{ nullptr }, mini_batch_size{ 0 }, mini_batch_images{ nullptr }, mini_batch_labels{ nullptr } {

    srand(time(NULL));
    load();
}

//Copy constructor
MNIST_DS::MNIST_DS(const MNIST_DS& data) 
    : train{ data.train }, n{ data.n }, m{ data.m }, magic_number{ data.magic_number }, number_of_items{ data.number_of_items }, image_dat_set{ nullptr },
    label_dat_set{ nullptr }, mini_batch_size{ data.mini_batch_size }, mini_batch_images{ nullptr }, mini_batch_labels{ nullptr }{

    image_dat_set = new Matrix[number_of_items];
    label_dat_set = new unsigned char[number_of_items];
    mini_batch_images = new Matrix[mini_batch_size];
    mini_batch_labels = new unsigned char[mini_batch_size];

    if (NULL != image_dat_set && NULL != label_dat_set && NULL != mini_batch_images && NULL != mini_batch_labels) {
        for (size_t i{ 0 }; i < number_of_items; i++) {
            image_dat_set[i] = data.image_dat_set[i];
            label_dat_set[i] = data.label_dat_set[i];
        }
        for (size_t i{ 0 }; i < mini_batch_size; i++) {
            mini_batch_images[i] = data.mini_batch_images[i];
            mini_batch_labels[i] = data.mini_batch_labels[i];
        }
    }
}

//Destructor
MNIST_DS::~MNIST_DS() {
    delete[] image_dat_set;
    delete[] label_dat_set;
    delete[] mini_batch_images;
    delete[] mini_batch_labels;
}

//Load data
void MNIST_DS::load() {
    string nameImagesFile{};
    string nameLabelsFile{};

    if (train) {
        nameImagesFile = "train-images.idx3-ubyte";
        nameLabelsFile = "train-labels.idx1-ubyte";
    }
    else {
        nameImagesFile = "t10k-images.idx3-ubyte";
        nameLabelsFile = "t10k-labels.idx1-ubyte";
    }

    //Read images
    ifstream inFile{ nameImagesFile, ios::in | ios::binary | ios::ate };

    if (!inFile) {
        cerr << "Error trying to open the file!\n" << endl;
        exit(1);
    }

    ifstream::pos_type size = inFile.tellg();
    inFile.seekg(0, ios::beg);

    inFile.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
    magic_number = reverseInt(magic_number);
    inFile.read(reinterpret_cast<char*>(&number_of_items), sizeof(int));
    number_of_items = reverseInt(number_of_items);
    inFile.read(reinterpret_cast<char*>(&n), sizeof(int));
    n = reverseInt(n);
    inFile.read(reinterpret_cast<char*>(&m), sizeof(int));
    m = reverseInt(m);

    unsigned char pixel{ 0 };
    Matrix image{ n, m };
    image_dat_set = new Matrix[number_of_items];

    if (NULL != image_dat_set) {
        cout << "Loading data set of " << ((train) ? "training" : "test") << " images..." << endl;
        for (size_t i{ 0 }; i < number_of_items; i++) {
            for (size_t j{ 0 }; j < n * m; j++) {
                inFile.read(reinterpret_cast<char*>(&pixel), sizeof(unsigned char));
                image[j] = pixel;
            }
            image_dat_set[i] = image;
        }
    }
    else {
        cerr << "Error trying to allocate dynamic memory!\n" << endl;
        exit(1);
    }

    cout << "Finished loading data set of " << ((train) ? "training" : "test") << " images..." << endl;
    inFile.close();

    //Read labels
    ifstream inLabels{ nameLabelsFile, ios::in | ios::binary | ios::ate };

    if (!inLabels) {
        cerr << "Error trying to open the file!\n" << endl;
        exit(1);
    }

    size = inLabels.tellg();
    inLabels.seekg(0, ios::beg);

    inLabels.read(reinterpret_cast<char*>(&magic_number), sizeof(int));
    magic_number = reverseInt(magic_number);
    inLabels.read(reinterpret_cast<char*>(&number_of_items), sizeof(int));
    number_of_items = reverseInt(number_of_items);

    unsigned char label{ 0 };
    label_dat_set = new unsigned char[number_of_items];

    if (NULL != label_dat_set) {
        cout << "\nLoading data set of " << ((train) ? "training" : "test") << " labels..." << endl;
        for (size_t i{ 0 }; i < number_of_items; i++) {
            inLabels.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));
            label_dat_set[i] = label;
        }
    }
    else {
        cerr << "Error trying to allocate dynamic memory!\n" << endl;
        exit(1);
    }

    cout << "Finished loading data set of " << ((train) ? "training" : "test") << " labels..." << endl;
    inLabels.close();
}

//Convert raw bytes to int
int MNIST_DS::reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

//Get image
Matrix MNIST_DS::getImage(int index) {
    return image_dat_set[index];
}

//Get label
int MNIST_DS::getLabel(int index) {
    return static_cast<int>(label_dat_set[index]);
}

//Create mini_batch
void MNIST_DS::randomSet(int n) {
    mini_batch_size = n;
    int* index_array = new int[mini_batch_size];

    //Optimize for different numbers
    for (size_t i{ 0 }; i < mini_batch_size; i++)
        index_array[i] = rand() % (number_of_items + 1);

    if (NULL != mini_batch_images)
        delete[] mini_batch_images;
    if (NULL != mini_batch_labels)
        delete[] mini_batch_labels;

    mini_batch_images = new Matrix[mini_batch_size];
    mini_batch_labels = new unsigned char[mini_batch_size];
    if (NULL != mini_batch_images && NULL != mini_batch_labels) {
        for (size_t i{ 0 }; i < mini_batch_size; i++) {
            mini_batch_images[i] = getImage(index_array[i]);
            mini_batch_labels[i] = getLabel(index_array[i]);
        }
    }
    else {
        cerr << "Error trying to allocate dynamic memory!\n" << endl;
        exit(1);
    }
}

Matrix* MNIST_DS::getMiniBatchImages() {
    return mini_batch_images;
}

unsigned char* MNIST_DS::getMiniBatchLabels() {
    return mini_batch_labels;
}