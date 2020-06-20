#include <iostream>
#include <string>
#include <random>
#include <iomanip>
#include <cstring>
#include <string>
#include <cmath>
#include <fstream>
#include "Matrix.h"

using std::cin;
using std::endl;
using std::cout;
using std::cerr;
using std::string;
using std::ofstream;
using std::ifstream;
using std::ios;

//Constructor
Matrix::Matrix()
	: n{ 0 }, m{ 0 }, matrix{ nullptr } {
}

//Constructor with matrix size
Matrix::Matrix(int rows, int columns)
	: Matrix() {

	//Configuration of random numbers with gaussian normal distribution
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<float> norm_dist{ 0.0f, 1.0f };

	matrix = new float[rows * columns];
	if (NULL != matrix) {
		n = rows;
		m = columns;
		for (size_t i{ 0 }; i < n * m; i++)
			matrix[i] = norm_dist(gen);
	}
}

//Copy constructor using deep copy because of raw pointer
Matrix::Matrix(const Matrix& mtx)
	: Matrix() {
	matrix = new float[mtx.n * mtx.m];
	if (NULL != matrix) {
		n = mtx.n;
		m = mtx.m;
		memcpy(matrix, mtx.matrix, n * m * sizeof(float));
	}
	else
		cout << "\tERROR - DEEP COPY FAILED!" << endl;
}

//Move constructor
Matrix::Matrix(Matrix&& mtx) 
	: matrix{ mtx.matrix }, n{ mtx.n }, m{ mtx.m }{
	mtx.matrix = nullptr;
}

//Destructor
Matrix::~Matrix() {
	if (n != 0)
		delete[] matrix;
}

//Add two matrices
Matrix Matrix::operator+(const Matrix& mtx) {
	if (m == mtx.m && n == mtx.n) {
		Matrix tempMatrix{ n, m };
		for (size_t i{ 0 }; i < n * m; i++)
			tempMatrix[i] = matrix[i] + mtx.matrix[i];
		return tempMatrix;
	}
	else {
		Matrix tempMatrix;
		cout << "\tMatrix dimensions must agree." << endl;
		return tempMatrix; //returns empty matrix
	}
}

//Substract two matrices
Matrix Matrix::operator-(const Matrix& mtx) {
	if (m == mtx.m && n == mtx.n) {
		Matrix tempMatrix{ n, m };
		for (size_t i{ 0 }; i < n * m; i++)
			tempMatrix[i] = matrix[i] - mtx.matrix[i];
		return tempMatrix;
	}
	else {
		Matrix tempMatrix;
		cout << "\tMatrix dimensions must agree." << endl;
		return tempMatrix; //returns empty matrix
	}
}

//Multiply two matrices
Matrix Matrix::operator*(const Matrix& mtx) {
	if (m == mtx.n) {
		Matrix tempMatrix{ n, mtx.m };
		float temp{ 0.0f };
		for (size_t k{ 0 }; k < n; k++) {
			for (size_t i{ 0 }; i < mtx.m; i++) {
				for (size_t j{ 0 }; j < mtx.n; j++)
					temp += matrix[k * m + j] * mtx.matrix[mtx.m * j + i];
				tempMatrix[k * mtx.m + i] = temp;
				temp = 0.0f;
			}
		}
		return tempMatrix;
	}
	else {
		Matrix tempMatrix;
		cout << "\tMatrix dimensions must agree." << endl;
		return tempMatrix; //returns empty matrix
	}
}

//Hadamard product
Matrix Matrix::operator^(const Matrix& mtx) {
	if (n == mtx.n && m == mtx.m) {
		Matrix tmp{ n, m };
		for (size_t i{ 0 }; i < n * m; i++)
			tmp[i] = matrix[i] * mtx.matrix[i];
		return tmp;
	}
	else {
		Matrix tmp;
		cout << "\nERROR WITH HADAMARD PRODUCT!" << endl;
		return tmp;
	}
}

//Transpose
Matrix Matrix::operator~() {
	Matrix temp{ m, n };
	for (size_t i{ 0 }; i < n; i++) {
		for (size_t j{ 0 }; j < m; j++)
			temp[j * n + i] = matrix[i * m + j];
	}
	return temp;
}

//Operator []
float& Matrix::operator[](int index) {
	if (index < n * m)
		return matrix[index];
	else
		throw std::string{ "\tIndex out of range!" };
}

//Assignment operator
Matrix& Matrix::operator=(const Matrix& rhs) {
	if (this == &rhs)
		return *this;
	else {
		delete[] matrix;
		n = 0;
		m = 0;
		matrix = new float[rhs.n * rhs.m];
		if (NULL != matrix) {
			n = rhs.n;
			m = rhs.m;
			memcpy(matrix, rhs.matrix, n * m * sizeof(float));
			return *this;
		}
	}
}

//Multiply matrix with scalar
Matrix operator*(const Matrix& mtx, const float& scalar) {
	Matrix tempMatrix{ mtx.n, mtx.m };
	for (size_t i{ 0 }; i < mtx.n * mtx.m; i++)
		tempMatrix[i] = mtx.matrix[i] * scalar;
	return tempMatrix;
}

//Multiply scalar with matrix
Matrix operator*(const float& scalar, const Matrix& mtx) {
	Matrix tempMatrix{ mtx.n, mtx.m };
	for (size_t i{ 0 }; i < mtx.n * mtx.m; i++)
		tempMatrix[i] = mtx.matrix[i] * scalar;
	return tempMatrix;
}

//Print matrix
std::ostream& operator<<(std::ostream& os, const Matrix& mtx) {
	os << std::fixed << std::setprecision(2);
	for (size_t i{ 0 }; i < mtx.n; i++) {
		for (size_t j{ 0 }; j < mtx.m; j++)
			os << std::setw(7) << mtx.matrix[i * mtx.m + j];
		os << endl;
	}
	return os;
}

//Set all values to zero
void Matrix::zeros() {
	memset(matrix, 0.0f, n * m * sizeof(float));
}

int Matrix::getSize() const {
	return n * m;
}

int Matrix::getRows() {
	return n;
}

int Matrix::getColumns() {
	return m;
}

//Write to binary file
void Matrix::saveMatrix(string name) {
	string nameFile{ name + ".dat" };
	ofstream outFile{ nameFile, ios::out | ios::binary };

	if (!outFile) {
		cerr << "Error trying to open the file!\n" << endl;
		exit(1);
	}

	outFile.write(reinterpret_cast<const char*>(&n), sizeof(int));
	outFile.write(reinterpret_cast<const char*>(&m), sizeof(int));
	outFile.write(reinterpret_cast<const char*>(matrix), n * m * sizeof(float));
	outFile.close();
}

//Read from binary file
void Matrix::readMatrix(string name) {
	string nameFile{ name + ".dat" };
	ifstream inFile{ nameFile, ios::in | ios::binary };

	if (!inFile) {
		cerr << "Error trying to open the file!\n" << endl;
		exit(1);
	}

	inFile.read(reinterpret_cast<char*>(&n), sizeof(int));
	inFile.read(reinterpret_cast<char*>(&m), sizeof(int));

	delete[] matrix;
	matrix = new float[n * m];

	if (matrix != NULL) {
		inFile.read(reinterpret_cast<char*>(matrix), n * m * sizeof(float));
		inFile.close();
	}
	else {
		cerr << "Error allocating dynamic memory!\n" << endl;
		inFile.close();
		exit(1);
	}
}