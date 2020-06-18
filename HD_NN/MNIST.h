#include <fstream>
#include <string>
#include "Matrix.h"
#pragma once

using std::string;
using std::ifstream;

class MNIST {
	string file;
	int n;
	int m;
	int magic_number;
	int number_of_items;
	int mini_batch_size;
	Matrix* image_dat_set;
	Matrix* label_dat_set;
	Matrix* mini_batch_imag;
	Matrix* mini_batch_label;
	int reverseInt(int i);

public:
	MNIST(string data);
	MNIST(const MNIST& data);
	~MNIST();

	void mini_batches(int n);
	void shuffle();
	int get_number_items();
	int get_mini_batch_size();
	Matrix getImage(int index);
	Matrix getLabel(int index);
	Matrix* getMiniBatchImages(int index);
	Matrix* getMiniBatchLabels(int index);
};