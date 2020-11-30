#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;

typedef struct {
	Matrix h_prev;
	Matrix c_prev;
	Matrix x_t;

	Matrix h_next;
	Matrix c_next;
	Matrix e_t;
	Matrix f_t;
	Matrix g_t;
	Matrix q_t;
} LSTM_cell_fwd_res;

typedef struct {
	vector<Matrix> h;
	vector<LSTM_cell_fwd_res> progress;
} LSTM_fwd_res;

typedef struct {
	vector<Matrix> theta;
	Matrix U;
	Matrix b2;
	vector<Matrix> h;
} LSTM_affine_fwd_res;

typedef struct {
	vector<Matrix> dh;
	Matrix dU;
} LSTM_affine_bwd_res;

typedef struct {
	Matrix dh_prev;
	Matrix dc_prev;
	map<string,Matrix> gradients;
} LSTM_cell_bwd_res;

typedef struct {
	map<string,Matrix> gradients_total;
} LSTM_bwd_res;

typedef struct {
	vector<double> lossses;
	map<string, Matrix> params;
} LSTM_training_res;

class LSTM {
public:
	LSTM(map<char, size_t> char_to_idx, map<size_t, char> idx_to_char, size_t vocab_size, size_t n_h = 256, size_t seq_len = 25);

	LSTM_training_res train(vector<char> data, size_t epochs, size_t batch_size = 512, double lr = 0.001);
	string sample(Matrix h_prev, Matrix c_prev, size_t size);

private:
	map<char, size_t> char_to_idx;
	map<size_t, char> idx_to_char;
	size_t vocab_size;
	size_t n_h;
	size_t seq_len;
	double beta1;
	double beta2;

	map<string, Matrix> params;
	map<string, Matrix> grads;
	map<string, Matrix> adam_params;

	double smooth_loss;
	default_random_engine sample_random_generator;

	// training methods
	LSTM_cell_fwd_res cell_forward(Matrix x, Matrix h_prev, Matrix c_prev);
	LSTM_cell_bwd_res cell_backward(Matrix dh_next, Matrix dc_next, LSTM_cell_fwd_res cache);
	LSTM_affine_fwd_res affine_forward(vector<Matrix> h, Matrix U, Matrix b2);
	LSTM_affine_bwd_res affine_backward(vector<Matrix> dtheta, Matrix U_cached, Matrix b2_cached, vector<Matrix> h_cached);
	LSTM_fwd_res forward(Matrix x, Matrix h_init, Matrix c_init); // x = matrix where every row is a hot-one encoding of a single letter
	LSTM_bwd_res backward(vector<Matrix> dh, vector<LSTM_cell_fwd_res> fwd_progress);

	// helper methods
	Matrix sigmoid(Matrix);
	Matrix softmax(Matrix);
	void clip_grads();
	void reset_grads();
	vector<double> one_hot_encode(size_t c);
};

#endif
