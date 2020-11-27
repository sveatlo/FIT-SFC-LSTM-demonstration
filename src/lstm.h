#ifndef LSTM_H
#define LSTM_H

#include "matrix.h"
#include <map>
#include <string>
#include <vector>

using namespace std;


typedef struct {
		Matrix y_hat;
		Matrix v;
		Matrix h;
		Matrix o;
		Matrix c;
		Matrix c_hat;
		Matrix i;
		Matrix f;
		Matrix z;
} LSTM_step_data ;

class LSTM {
public:
  LSTM(map<char, size_t> char_to_idx, map<size_t, char> idx_to_char,
       size_t vocab_size, size_t n_h, size_t seq_len, double beta1,
       double beta2);

  void train(vector<char> data, size_t epochs, double lr);
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

  Matrix sigmoid(Matrix);
  Matrix softmax(Matrix);
  void clip_grads();
  void reset_grads();
  void update_params(size_t batch_n);
  LSTM_step_data forward_step(vector<size_t> x, Matrix h_prev, Matrix c_prev);
  void backward_step();
  void forward_backward(vector<size_t> x_batch, vector<size_t> y_batch, Matrix h_prev, Matrix c_prev);
};



#endif
