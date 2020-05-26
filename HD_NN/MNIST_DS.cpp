#include "MNIST_DS.h"
#include "Matrix.h"
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>

using std::endl;
using std::cout;
using std::vector;
using std::string;
using std::ofstream;
using std::ifstream;
using std::ios;
using std::cerr;

//Constructor
MNIST_DS::MNIST_DS(bool flag)
    : train{ flag }, n{ 0 }, m{ 0 }, magic_number{ 0 }, number_of_items{ 0 }, mini_batch_size{ 0 } {

    image_dat_set = vector<Matrix>{};
    label_dat_set = vector<Matrix>{};
    mini_batch_imag = vector<vector<Matrix>>{};
    mini_batch_label = vector<vector<Matrix>>{};

    load();
}

//Copy constructor
MNIST_DS::MNIST_DS(const MNIST_DS& data) 
    : train{ data.train }, n{ data.n }, m{ data.m }, magic_number{ data.magic_number }, number_of_items{ data.number_of_items }, mini_batch_size{ data.mini_batch_size }{

    //image_dat_set = vector<Matrix>(number_of_items);
    //label_dat_set = vector<Matrix>(number_of_items);
    //mini_batch_imag = vector<vector<Matrix>>(number_of_items / mini_batch_size, vector<Matrix>(mini_batch_size));
    //mini_batch_label = vector<vector<Matrix>>(number_of_items / mini_batch_size, vector<Matrix>(mini_batch_size));

    image_dat_set = data.image_dat_set;
    label_dat_set = data.label_dat_set;
    mini_batch_imag = data.mini_batch_imag;
    mini_batch_label = data.mini_batch_label;

    //for (size_t i{ 0 }; i < number_of_items; i++) {
    //    image_dat_set.at(i) = data.image_dat_set.at(i);
    //    label_dat_set.at(i) = data.label_dat_set.at(i);
    //}

    //for (size_t i{ 0 }; i < number_of_items / mini_batch_size; i++) {
    //    for (size_t j{ 0 }; j < mini_batch_size; j++) {
    //        mini_batch_imag.at(i).at(j) = data.mini_batch_imag.at(i).at(j);
    //        mini_batch_label.at(i).at(j) = data.mini_batch_label.at(i).at(j);
    //    }
    //}
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
    //image_dat_set = vector<Matrix>(number_of_items);

    cout << "Loading data set of " << ((train) ? "training" : "test") << " images..." << endl;
    for (size_t i{ 0 }; i < number_of_items; i++) {
        for (size_t j{ 0 }; j < n * m; j++) {
            inFile.read(reinterpret_cast<char*>(&pixel), sizeof(unsigned char));
            image[j] = pixel;
        }
        image_dat_set.push_back(image);
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
    Matrix temp{ 1, 10 };
    //label_dat_set = new unsigned char[number_of_items];

    cout << "\nLoading data set of " << ((train) ? "training" : "test") << " labels..." << endl;
    for (size_t i{ 0 }; i < number_of_items; i++) {
        inLabels.read(reinterpret_cast<char*>(&label), sizeof(unsigned char));

        for (size_t j{ 0 }; j < 10; j++)
            temp[j] = (j == static_cast<int>(label)) ? 1 : 0;
        label_dat_set.push_back(temp);
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

//Get number of items
int MNIST_DS::get_number_items() {
    return number_of_items;
}

//Get image
Matrix MNIST_DS::getImage(int index) {
    return image_dat_set.at(index);
}

//Get label
Matrix MNIST_DS::getLabel(int index) {
    return label_dat_set.at(index);
}

void MNIST_DS::shuffle() {
    vector<int> indexes;
    indexes.reserve(number_of_items);

    for (size_t i{ 0 }; i < number_of_items; i++)
        indexes.push_back(i);
    std::random_shuffle(indexes.begin(), indexes.end());

    vector<Matrix> temp_image;
    temp_image.reserve(number_of_items);
    vector<Matrix> temp_label;
    temp_label.reserve(number_of_items);

    for (size_t i{ 0 }; i < number_of_items; i++) {
        temp_image.push_back(image_dat_set.at(indexes.at(i)));
        temp_label.push_back(label_dat_set.at(indexes.at(i)));
    }

    image_dat_set = temp_image;
    label_dat_set = temp_label;
}

void MNIST_DS::mini_batches(int n) {
    mini_batch_size = n;
    int index{ 0 };

    mini_batch_imag = vector<vector<Matrix>>(number_of_items / mini_batch_size, vector<Matrix>(mini_batch_size));
    mini_batch_label = vector<vector<Matrix>>(number_of_items / mini_batch_size, vector<Matrix>(mini_batch_size));

    for (size_t i{ 0 }; i < number_of_items / mini_batch_size; i++) {
        for (int j{ index }; j < index + mini_batch_size; j++) {
            mini_batch_imag.at(i).at(j % mini_batch_size) = image_dat_set.at(j);
            mini_batch_label.at(i).at(j % mini_batch_size) = label_dat_set.at(j);
        }
        index += mini_batch_size;
    }

    //vector<vector<Matrix>> temp_images = vector<vector<Matrix>>(number_of_items / mini_batch_size, vector<Matrix>(mini_batch_size));
    //vector<vector<Matrix>> temp_labels = vector<vector<Matrix>>(number_of_items / mini_batch_size, vector<Matrix>(mini_batch_size));

    //for (size_t i{ 0 }; i < number_of_items / mini_batch_size; i++) {
    //    for (int j{ index }; j < index + mini_batch_size; j++) {
    //        temp_images.at(i).at(j % mini_batch_size) = image_dat_set.at(j);
    //        temp_labels.at(i).at(j % mini_batch_size) = label_dat_set.at(j);
    //    }
    //    index += mini_batch_size;
    //}

    //mini_batch_imag = temp_images;
    //mini_batch_label = temp_labels;
}

int MNIST_DS::get_mini_batch_size() {
    return mini_batch_size;
}

vector<Matrix> MNIST_DS::getMiniBatchImages(int index) {
    return mini_batch_imag.at(index);
}

vector<Matrix> MNIST_DS::getMiniBatchLabels(int index) {
    return mini_batch_label.at(index);
}