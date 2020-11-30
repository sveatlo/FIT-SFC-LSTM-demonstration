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

Matrix::Matrix() {
	this->rows = 0;
	this->cols = 0;
	this->data = vector<double>(0);
}

Matrix::Matrix(vector<double> &m) {
	this->rows = 1;
	this->cols = m.size();
	this->data = m;
}

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

Matrix Matrix::randn(size_t rows, size_t cols) {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::normal_distribution<double> distribution(0,1);

	vector<double> d;
	for (size_t i = 0; i < rows*cols; i++) {
		d.push_back(distribution(generator));
	}

	Matrix rm = Matrix(d);
	rm.reshape(rows, cols);
	return rm;
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

Matrix Matrix::operator+(Matrix &B) {
	if (B.rows == 1 && this->rows != 1 && this->cols == B.cols) {
		return this->batch_column_add(B.data);
	}
	if (B.rows != 1 && this->rows == 1 && this->cols == B.cols) {
		return B.batch_column_add(this->data);
	}

	// if (B.cols == 1 && this->rows == B.rows) {
	//     return this->batch_row_add(B.data);
	// }

	assert(B.cols == this->cols && B.rows == this->rows);

	Matrix res(this->rows, B.cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < B.cols; j++) {
			res(i, j) = ((*this)(i, j)) + B(i, j);
		}
	}

	return res;
}

Matrix Matrix::operator-(Matrix &B) {
	assert(B.cols == this->cols && B.rows == this->rows);

	Matrix res(this->rows, B.cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < B.cols; j++) {
			res(i, j) = ((*this)(i, j)) - B(i, j);
		}
	}

	return res;
}

Matrix Matrix::operator*(Matrix &B) {
	assert(this->rows == B.rows && this->cols == B.cols);

	Matrix res(*this);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < B.cols; j++) {
			res(i, j) *= B(i, j);
		}
	}

	return res;

	// assert(this->cols == B.rows);
	//
	// Matrix res(this->rows, B.cols, 0);
	//
	// for (size_t i = 0; i < this->rows; i++) {
	//	 for (size_t j = 0; j < B.cols; j++) {
	//		 double tmp = 0.0;
	//		 for (size_t k = 0; k < this->cols; k++) {
	//			 tmp += (*this)(i, k) * B(k, j);
	//		 }
	//
	//		 res(i, j) = tmp;
	//	 }
	// }
	//
	// return res;
}

Matrix Matrix::operator/(Matrix &B) {
	assert(this->rows == B.rows && this->cols == B.cols);

	Matrix res(*this);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < B.cols; j++) {
			res(i, j) /= B(i, j);
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

Matrix Matrix::sqrt() {
	Matrix res(this->rows, this->cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			double v = (*this)(i, j);
			res(i, j) = std::sqrt(v);
		}
	}

	return res;
}

Matrix Matrix::pow(double p) {
	Matrix res(this->rows, this->cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			double v = (*this)(i, j);
			res(i, j) = std::pow(v, p);
		}
	}

	return res;
}

Matrix Matrix::tanh() {
	Matrix res(this->rows, this->cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			double v = (*this)(i, j);
			res(i, j) = std::tanh(v);
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

Matrix Matrix::dot(Matrix &B) {
	if (this->rows == 1 && B.rows == 1) {
		B = B.transpose();
	}

	assert(this->cols == B.rows);

	Matrix res(this->rows, B.cols, 0);
	for (size_t i = 0; i < this->rows; i++) {
		vector<double> row = this->row(i);
		for (size_t j = 0; j < B.cols; j++) {
			vector<double> col = B.column(j);
			res(i, j) = inner_product(row.begin(), row.end(), col.begin(), 0);
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

Matrix Matrix::sum_cols() {
	Matrix res(1, this->cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			res(0, j) += (*this)(i,j);
		}
	}

	return res;
}

Matrix Matrix::batch_column_add(vector<double> b) {
	assert(b.size() == this->cols);

	Matrix res = (*this); // copy current matrix

	for (size_t i = 0; i < this->rows; i++) {
		auto it_begin = this->data.begin() + i * this->cols;
		std::transform (it_begin, it_begin + this->cols, b.begin(), it_begin, std::plus<double>());
	}

	return res;
}

vector<double> Matrix::ravel() {
	return vector<double>(this->data.begin(), this->data.end());
}

void Matrix::reshape(size_t _rows, size_t _cols) {
	if (_rows * _cols != this->rows * this->cols) {
		throw "Incompatible shape";
	}

	this->rows = _rows;
	this->cols = _cols;
}

void Matrix::fill(double filler) {
	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			(*this)(i, j) = filler;
		}
	}
}

Matrix Matrix::hstack(Matrix &B) {
	assert(this->rows == B.rows);

	Matrix stacked(this->rows, this->cols + B.cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			stacked(i, j) = (*this)(i, j);
		}
		for (size_t j = 0; j < B.cols; j++) {
			stacked(i, j + this->cols) = B(i, j);
		}
	}

	return stacked;
}

Matrix Matrix::vstack(Matrix &B) {
	assert(this->cols == B.cols);

	Matrix stacked(this->rows + B.rows, this->cols, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			stacked(i, j) = (*this)(i, j);
		}
	}
	for (size_t i = 0; i < B.rows; i++) {
		for (size_t j = 0; j < B.cols; j++) {
			stacked(this->rows + i, j) = B(i, j);
		}
	}

	return stacked;
}

pair<size_t, size_t> Matrix::shape() {
	return make_pair(this->rows, this->cols);
}
size_t Matrix::rows_n() { return this->rows; }
size_t Matrix::cols_n() { return this->cols; }

vector<double> Matrix::row(size_t n) {
	auto start = this->data.begin() + n * this->cols;
	return vector<double>(start, start + this->cols);
}

void Matrix::row(size_t n, vector<double> r) {
	assert(r.size() == this->cols); // make sure the vector inserted is the correct length

	copy(r.begin(), r.end(), this->data.begin() + n * this->cols);
}

vector<double> Matrix::column(size_t n) {
	vector<double> col;

	for (size_t i = 0; i < this->rows; i++) {
		col.push_back((*this)(i, n));
	}

	return col;
}

// void Matrix::randomize(double lower_bound, double upper_bound) {
//	 std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
//	 std::default_random_engine re;
//
//	 for (size_t i = 0; i < this->rows; i++) {
//		 for (size_t j = 0; j < this->cols; j++) {
//			 (*this)(i, j) = unif(re);
//		 }
//	 }
// }

Matrix Matrix::transpose() {
	Matrix transposed(this->cols, this->rows, 0);

	for (size_t i = 0; i < this->rows; i++) {
		for (size_t j = 0; j < this->cols; j++) {
			transposed(j, i) = (*this)(i, j);
		}
	}

	return transposed;
}

void Matrix::print() const {
	cout << "[";
	for (unsigned i = 0; i < this->rows; i++) {
		if (i != 0) {
			cout << " ";
		}
		cout << "[ ";
		for (unsigned j = 0; j < this->cols; j++) {
			cout << (*this)(i, j) << " ";
		}
		cout << "]";
		if (i != this->rows - 1) {
			cout << "," << endl;
		}
	}
	cout << "]" << endl;
}
