#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;


class Param {
public:
	Param() {}
	Param(string name, Matrix<double> value) : v(value) {
		this->name = name;
		this->d = Matrix<double>(this->v.rows_n(), this->v.cols_n(), 0);
		this->m = Matrix<double>(this->v.rows_n(), this->v.cols_n(), 0);
	}

	string name;
	Matrix<double> v;
	Matrix<double> d;
	Matrix<double> m;
};

typedef struct {
	Matrix<double> y;
	Matrix<double> v;
	Matrix<double> h;
	Matrix<double> o;
	Matrix<double> c;
	Matrix<double> c_bar;
	Matrix<double> i;
	Matrix<double> f;
	Matrix<double> z;
} LSTM_step_data;

typedef struct {
	double loss;
	Matrix<double> h;
	Matrix<double> c;
} LSTM_forward_backward_return;

typedef struct {
	Matrix<double> dh_prev;
	Matrix<double> dc_prev;
} LSTM_backward_return;

typedef struct {
	vector<double> lossses;
	map<string, Param> params;
} LSTM_training_res;

class LSTM {
public:
	LSTM(map<char, unsigned> _char_to_idx, map<unsigned, char> _idx_to_char,  unsigned _vocab_size, unsigned _n_h = 100, unsigned _seq_len = 25, 	 double _beta1 = 0.9, double _beta2 = 0.999);

	LSTM_training_res train(vector<char> data, unsigned epochs, double lr);
	string sample(unsigned size, char seed = '\0');

private:
	map<char, unsigned> char_to_idx;
	map<unsigned, char> idx_to_char;
	unsigned vocab_size;
	unsigned n_h;
	unsigned seq_len;
	double beta1;
	double beta2;

	map<string, Param> params;

	double smooth_loss;
	default_random_engine sample_random_generator;

	Matrix<double> sigmoid(Matrix<double>);
	Matrix<double> softmax(Matrix<double>);
	Matrix<double> dsigmoid(Matrix<double>);
	Matrix<double> dtanh(Matrix<double>);
	void clip_grads();
	void reset_grads();
	void update_params(double lr);
	LSTM_step_data forward_step(Matrix<double> x, Matrix<double> h_prev, Matrix<double> c_prev);
	LSTM_backward_return backward_step(unsigned idx, Matrix<double> dh_next, Matrix<double> dc_next, Matrix<double> c_prev, Matrix<double> z, Matrix<double> f, Matrix<double> i, Matrix<double> c_bar, Matrix<double> c, Matrix<double> o, Matrix<double> h, Matrix<double> v, Matrix<double> y);
	LSTM_forward_backward_return forward_backward(vector<unsigned> x_batch, vector<unsigned> y_batch, Matrix<double> h_prev, Matrix<double> c_prev);
};

#endif
