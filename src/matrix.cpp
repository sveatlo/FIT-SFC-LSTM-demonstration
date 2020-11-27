#include "matrix.h"
#include <algorithm>
#include <cmath>
#include <float.h>
#include <iostream>
#include <iterator>
#include <random>
#include <tuple>
#include <vector>

using namespace std;

Matrix::Matrix() {}

Matrix::Matrix(const Matrix &m) {
  this->rows = m.rows;
  this->cols = m.cols;
  this->data = vector<double>(m.data);
}

Matrix::Matrix(size_t rows, size_t cols, double initial_value) {
  if (rows == 0 || cols == 0)
    throw "Invalid Matrix size";

  this->rows = rows;
  this->cols = cols;
  this->data = vector<double>(rows * cols, initial_value);
}

Matrix::~Matrix() {}

Matrix &Matrix::operator=(const Matrix &m) {
  this->rows = m.rows;
  this->cols = m.cols;
  this->data = vector<double>(m.data);

  return *this;
}

double &Matrix::operator()(const size_t &row, const size_t &col) {
  if (row >= this->rows || col >= this->cols)
    throw "Matrix subscript out of bounds";

  return this->data[this->cols * row + col];
}

double Matrix::operator()(const size_t &row, const size_t &col) const {
  if (row >= this->rows || col >= this->cols)
    throw "const Matrix subscript out of bounds";

  return this->data[this->cols * row + col];
}

/*
 * matrix operations
 */

Matrix Matrix::operator*(Matrix &B) {
  if (B.cols != this->rows) {
    throw "incompatible shapes";
  }

  Matrix res(this->rows, B.cols, 0);

  double tmp = 0.0;
  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < B.cols; j++) {
      tmp = 0.0;
      for (size_t k = 0; k < this->cols; k++) {
        tmp += (*this)(i, k) * B(k, j);
      }

      res(i, j) = tmp;
    }
  }

  return res;
}

/*
 * scalar operations
 */

Matrix Matrix::operator+(double c) {
  Matrix res(this->rows, this->cols, c);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      res(i, j) += (*this)(i, j);
    }
  }

  return res;
}

Matrix Matrix::operator-(double c) {
  Matrix res(this->rows, this->cols, 0);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      res(i, j) = (*this)(i, j) - c;
    }
  }

  return res;
}

Matrix Matrix::operator*(double c) {
  Matrix res(this->rows, this->cols, c);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      res(i, j) *= (*this)(i, j);
    }
  }

  return res;
}

Matrix Matrix::operator/(double c) {
  Matrix res(this->rows, this->cols, 0);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      res(i, j) = (*this)(i, j) / c;
    }
  }

  return res;
}

/*
 * HELPER FUNCTIONS
 */

Matrix Matrix::exp() {
  Matrix res(this->rows, this->cols, 0);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      double v = (*this)(i, j);
      res(i, j) = std::exp(v);
    }
  }

  return res;
}

Matrix Matrix::divides(double numerator) {
  Matrix res(this->rows, this->cols, 0);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      double v = (*this)(i, j);

      res(i, j) = numerator / v;
    }
  }

  return res;
}

Matrix Matrix::clip(double min, double max) {
  Matrix res(this->rows, this->cols, 0);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      double v = (*this)(i, j);

      if (v > max) {
        v = max;
      } else if (v < min) {
        v = min;
      }

      res(i, j) = v;
    }
  }

  return res;
}

double Matrix::max() {
  double max = -DBL_MAX;

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      double v = (*this)(i, j);
      if (v > max) {
        max = v;
      }
    }
  }

  return max;
}

double Matrix::sum() {
  double sum = 0;

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      sum += (*this)(i, j);
    }
  }

  return sum;
}

void Matrix::fill(double filler) {
  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      (*this)(i, j) = filler;
    }
  }
}

tuple<size_t, size_t> Matrix::shape() {
  return tuple<size_t, size_t>(this->rows, this->cols);
}
size_t Matrix::rows_n() { return this->rows; }
size_t Matrix::cols_n() { return this->cols; }

void Matrix::randomize(double lower_bound, double upper_bound) {
  std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
  std::default_random_engine re;

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      (*this)(i, j) = unif(re);
    }
  }
}

Matrix Matrix::transpose() {
  Matrix transposed(this->rows, this->cols, 0);

  for (size_t i = 0; i < this->rows; i++) {
    for (size_t j = 0; j < this->cols; j++) {
      transposed(i, j) = (*this)(j, i);
    }
  }

  return transposed;
}

void Matrix::print() const {
  cout << "[ " << endl;
  for (unsigned i = 0; i < this->rows; i++) {
    cout << "\t[ ";
    for (unsigned j = 0; j < this->cols; j++) {
      cout << "[" << (*this)(i, j) << "] ";
    }
    cout << "]" << endl;
  }
  cout << "]" << endl;
}
