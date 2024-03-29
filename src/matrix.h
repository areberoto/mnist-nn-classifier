/*
 *SPDX-License-Identifier: MIT License
 *Copyright (c) 2020,2022 Alberto Alvarez
*/

#pragma once
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

class Matrix {
  int n;         // number of rows
  int m;         // number of columns
  float *matrix; // Matrix elements

public:
  Matrix();
  Matrix(int, int);
  Matrix(const Matrix &mtx);
  ~Matrix();
  Matrix operator+(const Matrix &mtx);
  Matrix operator-(const Matrix &mtx);
  Matrix operator*(const Matrix &mtx);
  Matrix operator^(const Matrix &mtx);
  Matrix &operator=(const Matrix &rhs);
  Matrix operator~();
  float &operator[](int);
  friend Matrix operator*(const Matrix &mtx, const float &scalar);
  friend Matrix operator*(const float &scalar, const Matrix &mtx);
  friend std::ostream &operator<<(std::ostream &os, const Matrix &mtx);
  int getSize() const;
  int getRows();
  int getColumns();
  void zeros();
  void saveMatrix(std::string name);
  void readMatrix(std::string name);
};
