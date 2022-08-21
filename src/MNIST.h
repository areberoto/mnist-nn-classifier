#include <fstream>
#include <string>
#include "Matrix.h"
#pragma once

using std::string;
using std::ifstream;

class MNIST {
	string file;	//Flag that indicates file
	int n;			//Number of image rows (28)
	int m;			//Number of image columns (28)
	int magic_number;	   //2049 or 2051
	int number_of_items;   //60,000 or 10,000
	int mini_batch_size;   //User's decision
	Matrix* image_dat_set; //Images from MNIST
	Matrix* label_dat_set; //Labels from MNIST
	Matrix* mini_batch_imag; //Mini-batch images
	Matrix* mini_batch_label;//Mini-batch labels
	int reverseInt(int i); //Raw bytes to int

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