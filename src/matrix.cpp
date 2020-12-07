#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.h"
#include <algorithm>
#include <chrono>
#include <assert.h>
#include <cmath>
#include <float.h>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

using namespace std;

template<typename T>
Matrix<T>::Matrix() {}

template<typename T>
Matrix<T>::Matrix(vector<T> &m) {
	this->rows = 1;
	this->cols = m.size();
	this->data = m;
}

template<typename T>
Matrix<T>::Matrix(const Matrix &m) {
	this->rows = m.rows;
	this->cols = m.cols;
	this->data = vector<T>(m.data);
}

template<typename T>
Matrix<T>::Matrix(unsigned rows, unsigned cols, T initial_value) {
	if (rows == 0 || cols == 0)
		throw "Invalid Matrix<T> size";

	this->rows = rows;
	this->cols = cols;
	this->data = vector<T>(rows * cols, initial_value);
}

template<typename T>
Matrix<T>::~Matrix() {}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &m) {
	this->rows = m.rows;
	this->cols = m.cols;
	this->data = vector<T>(m.data);

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::randn(unsigned rows, unsigned cols) {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<T> distribution(0,1);

	vector<T> d;
	for (unsigned i = 0; i < rows*cols; i++) {
		d.push_back(distribution(generator));
	}

	Matrix<T> rm = Matrix(d);
	rm.reshape(rows, cols);
	return rm;
}


template<typename T>
T& Matrix<T>::operator()(const unsigned &row, const unsigned &col) {
	if (row >= this->rows || col >= this->cols)
		throw "Matrix<T> subscript out of bounds";

	return this->data[this->cols * row + col];
}

template<typename T>
T Matrix<T>::operator()(const unsigned &row, const unsigned &col) const {
	if (row >= this->rows || col >= this->cols)
		throw "const Matrix<T> subscript out of bounds";

	return this->data[this->cols * row + col];
}

/*
 * matrix operations
 */

template<typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T>& rhs) {
	assert(rhs.cols == this->cols && rhs.rows == this->rows);

	Matrix<T> res(this->rows, rhs.cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < rhs.cols; j++) {
			res(i, j) = ((*this)(i, j)) + rhs(i, j);
		}
	}

	return res;
}

template<typename T>
Matrix<T>& Matrix<T>::operator+=(Matrix<T> rhs) {
	for (unsigned i=0; i<this->rows; i++) {
		for (unsigned j=0; j<this->cols; j++) {
			(*this)(i,j) += rhs(i,j);
		}
	}

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(Matrix<T>& rhs) {
	assert(rhs.cols == this->cols && rhs.rows == this->rows);

	Matrix<T> res(this->rows, rhs.cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < rhs.cols; j++) {
			res(i, j) = ((*this)(i, j)) - rhs(i, j);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator-=(Matrix<T> rhs) {
	for (unsigned i=0; i<this->rows; i++) {
		for (unsigned j=0; j<this->cols; j++) {
			(*this)(i,j) -= rhs(i,j);
		}
	}

	return *this;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T>& rhs) {
	assert(this->rows == rhs.rows && this->cols == rhs.cols);

	Matrix<T> res(*this);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < rhs.cols; j++) {
			res(i, j) *= rhs(i, j);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(Matrix<T>& rhs) {
	assert(this->rows == rhs.rows && this->cols == rhs.cols);

	Matrix<T> res(*this);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < rhs.cols; j++) {
			res(i, j) /= rhs(i, j);
		}
	}

	return res;
}

/*
 * scalar operations
 */

template<typename T>
Matrix<T> Matrix<T>::operator+(T c) {
	Matrix<T> res(this->rows, this->cols, c);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			res(i, j) += (*this)(i, j);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(T c) {
	Matrix<T> res(this->rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			res(i, j) = (*this)(i, j) - c;
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(T c) {
	Matrix<T> res(this->rows, this->cols, c);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			res(i, j) *= (*this)(i, j);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(T c) {
	Matrix<T> res(this->rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			res(i, j) = (*this)(i, j) / c;
		}
	}

	return res;
}

/*
 * HELPER FUNCTIONS
 */

template<typename T>
Matrix<T> Matrix<T>::exp() {
	Matrix<T> res(this->rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			T v = (*this)(i, j);
			res(i, j) = std::exp(v);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::sqrt() {
	Matrix<T> res(this->rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			T v = (*this)(i, j);
			assert(v == v);
			res(i, j) = std::sqrt(v);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::pow(T p) {
	Matrix<T> res(this->rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			T v = (*this)(i, j);
			res(i, j) = std::pow(v, p);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::tanh() {
	Matrix<T> res(this->rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			T v = (*this)(i, j);
			res(i, j) = std::tanh(v);
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::divides(T numerator) {
	Matrix<T> res(this->rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			T v = (*this)(i, j);

			res(i, j) = numerator / v;
		}
	}

	return res;
}

template<typename T>
Matrix<T> Matrix<T>::dot(Matrix & rhs) {
	if (this->rows == 1 && rhs.rows == 1) {
		rhs = rhs.transpose();
	}

	assert(this->cols == rhs.rows);

	Matrix<T> result(this->rows, rhs.cols, 0.0);

	for(unsigned i = 0; i < this->rows; ++i)
        for(unsigned j = 0; j < rhs.cols; ++j)
            for(unsigned k = 0; k < this->cols; ++k)
				result(i,j) += (*this)(i,k) * rhs(k,j);

	return result;

}

template<typename T>
void Matrix<T>::clip(T min, T max) {
	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			T v = (*this)(i, j);

			if (v > max) {
				v = max;
			} else if (v < min) {
				v = min;
			}

			(*this)(i, j) = v;
		}
	}
}

template<typename T>
T Matrix<T>::max() {
	T max = -LDBL_MAX;

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			T v = (*this)(i, j);
			if (v > max) {
				max = v;
			}
		}
	}

	return max;
}

template<typename T>
T Matrix<T>::sum() {
	T sum = 0;

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			sum += (*this)(i, j);
		}
	}

	return sum;
}

template<typename T>
vector<T> Matrix<T>::ravel() {
	return vector<T>(this->data.begin(), this->data.end());
}

template<typename T>
void Matrix<T>::reshape(unsigned _rows, unsigned _cols) {
	if (_rows * _cols != this->rows * this->cols) {
		throw "Incompatible shape";
	}

	this->rows = _rows;
	this->cols = _cols;
}

template<typename T>
void Matrix<T>::fill(T filler) {
	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			(*this)(i, j) = filler;
		}
	}
}

template<typename T>
Matrix<T> Matrix<T>::hstack(Matrix<T>& rhs) {
	assert(this->rows == rhs.rows);

	Matrix<T> stacked(this->rows, this->cols + rhs.cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			stacked(i, j) = (*this)(i, j);
		}
		for (unsigned j = 0; j < rhs.cols; j++) {
			stacked(i, j + this->cols) = rhs(i, j);
		}
	}

	return stacked;
}

template<typename T>
Matrix<T> Matrix<T>::vstack(Matrix<T>& rhs) {
	assert(this->cols == rhs.cols);

	Matrix<T> stacked(this->rows + rhs.rows, this->cols, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			stacked(i, j) = (*this)(i, j);
		}
	}
	for (unsigned i = 0; i < rhs.rows; i++) {
		for (unsigned j = 0; j < rhs.cols; j++) {
			stacked(this->rows + i, j) = rhs(i, j);
		}
	}

	return stacked;
}

template<typename T>
tuple<unsigned, unsigned> Matrix<T>::shape() {
	return tuple<unsigned, unsigned>(this->rows, this->cols);
}
template<typename T>
unsigned Matrix<T>::rows_n() { return this->rows; }
template<typename T>
unsigned Matrix<T>::cols_n() { return this->cols; }

template<typename T>
vector<T> Matrix<T>::row(unsigned n) {
	auto start = this->data.begin() + n * this->cols;
	return vector<T>(start, start + this->cols);
}

template<typename T>
vector<T> Matrix<T>::column(unsigned n) {
	vector<T> col;

	for (unsigned i = 0; i < this->rows; i++) {
		col.push_back((*this)(i, n));
	}

	return col;
}

template<typename T>
Matrix<T> Matrix<T>::transpose() {
	Matrix<T> transposed(this->cols, this->rows, 0);

	for (unsigned i = 0; i < this->rows; i++) {
		for (unsigned j = 0; j < this->cols; j++) {
			transposed(j, i) = (*this)(i, j);
		}
	}

	return transposed;
}

template<typename T>
void Matrix<T>::print() const {
	cout << "[";
	for (unsigned i = 0; i < this->rows; i++) {
		if (i != 0) {
			cout << " ";
		}
		cout << "[ ";
		for (unsigned j = 0; j < this->cols; j++) {
			cout.precision(8);
			cout << (*this)(i, j) << " ";
		}
		cout << "]";
		if (i != this->rows - 1) {
			cout << "," << endl;
		}
	}
	cout << "]" << endl;
}

#endif
