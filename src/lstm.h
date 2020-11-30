#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;

typedef struct {
	Matrix<double> y_hat;
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
	map<string, Matrix<double>> params;
} LSTM_training_res;

class LSTM {
public:
	LSTM(map<char, size_t> _char_to_idx, map<size_t, char> _idx_to_char,
			 size_t _vocab_size, size_t _n_h = 100, size_t _seq_len = 25,
			 double _beta1 = 0.9, double _beta2 = 0.999);

	LSTM_training_res train(vector<char> data,
																									size_t epochs, double lr);
	string sample(Matrix<double> h_prev, Matrix<double> c_prev, size_t size);

private:
	map<char, size_t> char_to_idx;
	map<size_t, char> idx_to_char;
	size_t vocab_size;
	size_t n_h;
	size_t seq_len;
	double beta1;
	double beta2;

	map<string, Matrix<double>> params;
	map<string, Matrix<double>> grads;
	map<string, Matrix<double>> adam_params;

	double smooth_loss;
	default_random_engine sample_random_generator;

	Matrix<double> sigmoid(Matrix<double>);
	Matrix<double> softmax(Matrix<double>);
	void clip_grads();
	void reset_grads();
	void update_params(double lr, size_t batch_n);
	LSTM_step_data forward_step(Matrix<double> x, Matrix<double> h_prev, Matrix<double> c_prev);
	LSTM_backward_return backward_step(size_t y, Matrix<double> y_hat, Matrix<double> dh_next,
																		 Matrix<double> dc_next, Matrix<double> c_prev, Matrix<double> z,
																		 Matrix<double> f, Matrix<double> i, Matrix<double> c_bar, Matrix<double> c,
																		 Matrix<double> o, Matrix<double> h);
	LSTM_forward_backward_return forward_backward(vector<size_t> x_batch,
																								vector<size_t> y_batch,
																								Matrix<double> h_prev, Matrix<double> c_prev);
};

#endif
