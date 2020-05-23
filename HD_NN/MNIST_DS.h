#include<fstream>
#include<string>
#pragma once

using std::string;
using std::ifstream;

class MNIST_DS {
	bool train;
	int n;
	int m;
	int magic_number;
	int number_of_items;
	int mini_batch_size;
	Matrix* image_dat_set;
	unsigned char* label_dat_set;
	Matrix* mini_batch_images;
	unsigned char* mini_batch_labels;
	void load();
	int reverseInt(int i);

public:
	MNIST_DS(bool flag);
	MNIST_DS(const MNIST_DS& data);
	~MNIST_DS();

	void randomSet(int n);
	Matrix getImage(int index);
	int getLabel(int index);
	Matrix* getMiniBatchImages();
	unsigned char* getMiniBatchLabels();
};