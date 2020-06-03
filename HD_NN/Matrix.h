#pragma once

class Matrix {
	int n;		//number of rows
	int m;		//number of columns
	float* matrix;

public:
	Matrix();
	Matrix(int, int);
	Matrix(const Matrix& mtx);
	~Matrix();
	Matrix operator+(const Matrix& mtx);
	friend Matrix operator+(const Matrix& mtx, float scalar);
	friend Matrix operator+(float scalar, const Matrix& mtx);
	Matrix operator-(const Matrix& mtx);
	//friend Matrix operator-(const Matrix& mtx, float scalar);
	//friend Matrix operator-(float scalar, const Matrix& mtx);
	Matrix operator*(const Matrix& mtx);
	friend Matrix operator*(const Matrix& mtx, float scalar);
	friend Matrix operator*(float scalar, const Matrix& mtx);
	Matrix dot(const Matrix& mtx);
	float operator!();
	float& operator[](int);
	Matrix& operator=(const Matrix& rhs);
	int getSize();
	void zeros();
	void transpose();
	Matrix hadamard(const Matrix& mtx);
	friend std::ostream& operator<<(std::ostream& os, const Matrix& mtx);
};