#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;

typedef struct {
  Matrix y_hat;
  Matrix y;
  Matrix h;
  Matrix o;
  Matrix c;
  Matrix c_bar;
  Matrix i;
  Matrix f;
  Matrix z;
  Matrix c_prev;
  Matrix h_prev;
} LSTM_cell_data;

typedef struct {
  double loss;
  Matrix h;
  Matrix c;
} LSTM_optimization_res;

typedef struct {
  Matrix dh_prev;
  Matrix dc_prev;
  Matrix dWf;
  Matrix dbf;
  Matrix dWi;
  Matrix dbi;
  Matrix dWc;
  Matrix dbc;
  Matrix dWo;
  Matrix dbo;
  Matrix dWy;
  Matrix dby;
} LSTM_backward_return;

typedef struct {
	vector<double> lossses;
	map<string, Matrix> params;
} LSTM_training_res;

class LSTM {
public:
  LSTM(map<char, size_t> _char_to_idx, map<size_t, char> _idx_to_char,
       size_t _vocab_size, size_t _n_h = 100, size_t _seq_len = 25,
       double _beta1 = 0.9, double _beta2 = 0.999);

  LSTM_training_res train(vector<char> data,
                                                  size_t epochs, double lr);
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

  Matrix sigmoid(Matrix);
  Matrix softmax(Matrix);
  Matrix one_hot_encode(size_t);
  void clip_grads();
  void reset_grads();
  void update_params(double lr);
  LSTM_cell_data cell_forward(Matrix x, Matrix h_prev, Matrix c_prev);
  LSTM_backward_return cell_backward(Matrix dh_next, Matrix dc_next, size_t char_idx, LSTM_cell_data cd);
  LSTM_optimization_res optimize(vector<size_t> x_batch,
                                                vector<size_t> y_batch,
                                                Matrix h_prev, Matrix c_prev, double lr);
};

#endif
