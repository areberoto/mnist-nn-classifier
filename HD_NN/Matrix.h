#pragma once

class Matrix {
	int n;		//number of rows
	int m;		//number of columns
	double* matrix;

public:
	Matrix();
	Matrix(int, int);
	Matrix(const Matrix& mtx);
	~Matrix();
	Matrix operator+(const Matrix& mtx);
	friend Matrix operator+(const Matrix& mtx, double scalar);
	friend Matrix operator+(double scalar, const Matrix& mtx);
	Matrix operator-(const Matrix& mtx);
	friend Matrix operator-(const Matrix& mtx, double scalar);
	friend Matrix operator-(double scalar, const Matrix& mtx);
	Matrix operator*(const Matrix& mtx);
	friend Matrix operator*(const Matrix& mtx, double scalar);
	friend Matrix operator*(double scalar, const Matrix& mtx);
	Matrix dot(const Matrix& mtx);
	double operator!();
	double& operator[](int);
	Matrix& operator=(const Matrix& rhs);
	int getSize();
	void zeros();
	Matrix hadamard(const Matrix& mtx);
	friend std::ostream& operator<<(std::ostream& os, const Matrix& mtx);
};