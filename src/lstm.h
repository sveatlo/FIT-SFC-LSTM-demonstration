#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <map>
#include <string>
#include <vector>

using namespace std;

class LSTM {
public:
  LSTM(map<char, size_t> char_to_idx, map<size_t, char> idx_to_char,
       size_t vocab_size, size_t n_h, size_t seq_len);
  LSTM(map<char, size_t> char_to_idx, map<size_t, char> idx_to_char,
       size_t vocab_size, size_t n_h, size_t seq_len, double beta1,
       double beta2);

  void train(vector<char> data, size_t epochs, double lr);
  string sample(Matrix h_prev, Matrix c_prev, size_t size);
  void set_seq();
  void print_debug();

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

  Matrix sigmoid(Matrix);
  Matrix softmax(Matrix);
  void clip_grads();
  void reset_grads();
  void update_params(size_t batch_n);
  void forward_step(vector<char> x, Matrix h_prev, Matrix c_prev);
  void backward_step();
  void forward_backward(vector<char> x_batch, vector<char> y_batch, Matrix h_prev, Matrix c_prev);
};

#endif
