#ifndef MATRIX_H
#define MATRIX_H

#include <tuple>
#include <vector>

using namespace std;

class Matrix {
public:
  Matrix();
  Matrix(size_t rows, size_t cols, double initial_value);
  ~Matrix();                         // Destructor
  Matrix(const Matrix &);            // Copy constructor
  Matrix &operator=(const Matrix &); // Assignment operator

  // matrix operations
  Matrix operator+(Matrix &);
  Matrix operator-(Matrix &);
  Matrix operator*(Matrix &);

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
  Matrix clip(double, double);
  double max();
  double sum();
  Matrix divides(double numerator);

  // helpers
  void fill(double);
  Matrix transpose();
  void randomize(double, double);
  void print() const;
  tuple<size_t, size_t> shape();
  size_t rows_n();
  size_t cols_n();

private:
  size_t rows;
  size_t cols;
  vector<double> data;
};

#endif
