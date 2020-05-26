#include <fstream>
#include <string>
#include <vector>
#include "Matrix.h"
#pragma once

using std::string;
using std::ifstream;
using std::vector;

class MNIST_DS {
	bool train;
	int n;
	int m;
	int magic_number;
	int number_of_items;
	int mini_batch_size;
	vector<Matrix> image_dat_set;
	vector<unsigned char> label_dat_set;
	vector<Matrix> mini_batch_imag;
	vector<unsigned char> mini_batch_label;
	void load();
	int reverseInt(int i);

public:
	MNIST_DS(bool flag);
	MNIST_DS(const MNIST_DS& data);

	void mini_batches(int mini_batch_size);
	void shuffle();
	int get_number_items();
	Matrix getImage(int index);
	int getLabel(int index);
};