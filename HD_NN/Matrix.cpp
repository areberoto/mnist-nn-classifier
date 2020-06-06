#include <iostream>
#include <string>
#include <random>
#include <iomanip>
#include <cmath>
#include "Matrix.h"

using std::cin;
using std::endl;
using std::cout;

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
		for (size_t i{ 0 }; i < n * m; i++)
			matrix[i] = mtx.matrix[i];
	}
	else
		cout << "\tERROR - DEEP COPY FAILED!" << endl;
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
			for (size_t i{ 0 }; i < n * m; i++)
				matrix[i] = rhs.matrix[i];
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
	for (size_t i{ 0 }; i < m * n; i++)
		matrix[i] = 0.0f;
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