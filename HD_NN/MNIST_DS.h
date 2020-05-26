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
	vector<Matrix> label_dat_set;
	vector<vector<Matrix>> mini_batch_imag;
	vector<vector<Matrix>> mini_batch_label;
	void load();
	int reverseInt(int i);

public:
	MNIST_DS(bool flag);
	MNIST_DS(const MNIST_DS& data);

	void mini_batches(int n);
	void shuffle();
	int get_number_items();
	int get_mini_batch_size();
	Matrix getImage(int index);
	Matrix getLabel(int index);
	vector<Matrix> getMiniBatchImages(int index);
	vector<Matrix> getMiniBatchLabels(int index);
};