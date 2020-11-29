#ifndef MATRIX_H
#define MATRIX_H

#include <tuple>
#include <vector>
#include <random>

using namespace std;

class Matrix {
public:
  Matrix();
  Matrix(vector<double> &);
  Matrix(size_t rows, size_t cols, double initial_value);
  ~Matrix();                         // Destructor
  Matrix(const Matrix &);            // Copy constructor
  Matrix &operator=(const Matrix &); // Assignment operator

  static Matrix randn(size_t rows, size_t cols);

  // matrix operations
  Matrix operator+(Matrix &);
  Matrix operator-(Matrix &);
  Matrix operator*(Matrix &);
  Matrix operator/(Matrix &);

  // scalar Operations
  Matrix operator+(double);
  Matrix operator-(double);
  Matrix operator*(double);
  Matrix operator/(double);

  // accessors
  double &operator()(const size_t &, const size_t &);
  double operator()(const size_t &, const size_t &) const;

  // arithmetic helpers
  Matrix exp();
  Matrix sqrt();
  Matrix pow(double);
  Matrix tanh();
  Matrix dot(Matrix &);
  Matrix divides(double numerator);
  Matrix clip(double, double);
  double max();
  double sum();

  // helpers
  vector<double> ravel();
  void reshape(size_t rows, size_t cols);
  void fill(double);
  Matrix hstack(Matrix &);
  Matrix vstack(Matrix &);
  Matrix transpose();
  tuple<size_t, size_t> shape();
  size_t rows_n();
  size_t cols_n();
  vector<double> row(size_t n);
  vector<double> column(size_t n);
  void print() const;

private:
  size_t rows;
  size_t cols;
  vector<double> data;
};

#endif
