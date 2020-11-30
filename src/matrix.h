#ifndef MATRIX_H
#define MATRIX_H

#include <tuple>
#include <vector>
#include <random>

using namespace std;

template <typename T>
class Matrix {
public:
	Matrix();
	Matrix(vector<T> &);
	Matrix(unsigned rows, unsigned cols, T initial_value);
	~Matrix();												 // Destructor
	Matrix(const Matrix &);						// Copy constructor
	Matrix &operator=(const Matrix &); // Assignment operator

	static Matrix randn(unsigned rows, unsigned cols);

	// matrix operations
	Matrix operator+(Matrix &);
	Matrix& operator+=(Matrix);
	Matrix operator-(Matrix &);
	Matrix operator-=(Matrix);
	Matrix operator*(Matrix &);
	Matrix& operator*=(Matrix &);
	Matrix operator/(Matrix &);

	// scalar Operations
	Matrix operator+(T);
	Matrix operator-(T);
	Matrix operator*(T);
	Matrix operator/(T);

	// accessors
	T &operator()(const unsigned &, const unsigned &);
	T operator()(const unsigned &, const unsigned &) const;

	// arithmetic helpers
	Matrix exp();
	Matrix sqrt();
	Matrix pow(T);
	Matrix tanh();
	Matrix dot(Matrix &);
	Matrix divides(T numerator);
	Matrix clip(T, T);
	T max();
	T sum();

	// helpers
	vector<T> ravel();
	void reshape(unsigned rows, unsigned cols);
	void fill(T);
	Matrix hstack(Matrix &);
	Matrix vstack(Matrix &);
	Matrix transpose();
	tuple<unsigned, unsigned> shape();
	unsigned rows_n();
	unsigned cols_n();
	vector<T> row(unsigned n);
	vector<T> column(unsigned n);
	void print() const;

private:
	unsigned rows;
	unsigned cols;
	vector<T> data;
};

#include "matrix.cpp"

#endif
