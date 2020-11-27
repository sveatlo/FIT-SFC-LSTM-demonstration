#include "lstm.h"
#include "matrix.h"
#include <iostream>
#include <map>
#include <math.h>
#include <string>

using namespace std;

LSTM::LSTM(map<char, size_t> _char_to_idx, map<size_t, char> _idx_to_char,
           size_t _vocab_size, size_t _n_h, size_t _seq_len, double _beta1 = 0.9,
           double _beta2 = 0.999)
    : char_to_idx(_char_to_idx), idx_to_char(_idx_to_char),
      vocab_size(_vocab_size), n_h(_n_h), seq_len(_seq_len), beta1(_beta1),
      beta2(_beta2) {
  double std = 1.0 / sqrt(this->vocab_size + this->n_h);

  // forget gate
  Matrix wf = Matrix(this->n_h, this->n_h + this->vocab_size, 0);
  wf.randomize(-1, 1);
  Matrix bf = Matrix(this->n_h, 1, 1);
  this->params.insert(make_pair("Wf", wf * std));
  this->params.insert(make_pair("bf", bf));

  // input gate
  Matrix wi(this->n_h, this->n_h + this->vocab_size, 0);
  wi.randomize(-1, 1);
  Matrix bi(this->n_h, 1, 1);
  this->params.insert(make_pair("Wi", wi * std));
  this->params.insert(make_pair("bi", bi));

  // cell state gate
  Matrix wc(this->n_h, this->n_h + this->vocab_size, 0);
  wc.randomize(-1, 1);
  Matrix bc(this->n_h, 1, 1);
  this->params.insert(make_pair("Wc", wc * std));
  this->params.insert(make_pair("bc", bc));

  // output gate
  Matrix wo(this->n_h, this->n_h + this->vocab_size, 0);
  wo.randomize(-1, 1);
  Matrix bo(this->n_h, 1, 1);
  this->params.insert(make_pair("Wo", wo * std));
  this->params.insert(make_pair("bo", bo));

  for (auto const &item : this->params) {
    string param_name = item.first;
    Matrix param_matrix = item.second;

    this->grads["d" + param_name] =
        Matrix(param_matrix.rows_n(), param_matrix.cols_n(), 0);
    this->adam_params["y" + param_name] =
        Matrix(param_matrix.rows_n(), param_matrix.cols_n(), 0);
    this->adam_params["m" + param_name] =
        Matrix(param_matrix.rows_n(), param_matrix.cols_n(), 0);
  }

  this->smooth_loss = -1 * log(1.0f / this->vocab_size) * this->seq_len;
}

Matrix LSTM::sigmoid(Matrix x) { return ((x * -1).exp() + 1).divides(1); }

Matrix LSTM::softmax(Matrix x) {
  Matrix e_x = (x - x.max());

  return e_x / e_x.sum();
}

void LSTM::clip_grads() {
  for (auto &item : this->grads) {
    item.second.clip(-5, 5);
  }
}

void LSTM::reset_grads() {
  for (auto &item : this->grads) {
    item.second.fill(0);
  }
}

void LSTM::update_params(size_t batch_n) {
  for (auto &item : this->params) {
    string key = item.first;

	auto mtmp1 = (this->grads["d" + key] * (1 - this->beta1));
	auto mtmp2 = (this->grads["d" + key] * (1 - this->beta2));
    this->adam_params["m" + key] = (this->adam_params["m" + key] * this->beta1) + mtmp1;
	this->adam_params["v" + key] = (this->adam_params["m" + key] * this->beta2) + mtmp2;
  }
}

LSTM_step_data LSTM::forward_step(vector<size_t> _x, Matrix h_prev, Matrix c_prev) {
	vector<double> __x;
	for(size_t x_n : _x) {
		__x.push_back(static_cast<double>(x_n));
	}
	Matrix x(__x);

	Matrix z = x.vstack(h_prev);

	Matrix f = this->sigmoid(this->params["Wf"].dot(z) + this->params["bf"]);
	Matrix i = this->sigmoid(this->params["Wi"].dot(z) + this->params["bi"]);
	Matrix c_hat = this->sigmoid(this->params["Wc"].dot(z) + this->params["bc"]);
	Matrix o = this->sigmoid(this->params["Wo"].dot(z) + this->params["bo"]);

	Matrix ctmp = i * c_hat;
	Matrix c = f * c_prev + ctmp;
	Matrix h = c.tanh() * o;
	Matrix v = this->sigmoid(this->params["Wv"].dot(h) + this->params["bv"]);
	Matrix y_hat = this->softmax(v);

	LSTM_step_data step_data = {
		.y_hat = y_hat,
		.v = v,
		.h = h,
		.o = o,
		.c = c,
		.c_hat = c_hat,
		.i = i,
		.f = f,
		.z = z,
	};


	return step_data;
}

void LSTM::backward_step() {}

void LSTM::forward_backward(vector<size_t> x_batch, vector<size_t> y_batch,
                            Matrix h_prev, Matrix c_prev) {}

void LSTM::train(vector<char> X, size_t epochs, double lr) {
  vector<double> losses;

  int num_batches = X.size() / this->seq_len;
  vector<char> X_trimmed(X.begin(), X.begin() + num_batches * this->seq_len);

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    Matrix h_prev(this->n_h, 1, 0);
    Matrix c_prev(this->n_h, 1, 0);

    for (size_t i = 0; i < X_trimmed.size(); i += this->seq_len) {
      // prepare data
      vector<size_t> x_batch, y_batch;
      for (size_t j = i; j < i + this->seq_len; j++) {
        x_batch.push_back(this->char_to_idx[j]);
      }
      for (size_t j = i + 1; j < i + this->seq_len + 1; j++) {
        y_batch.push_back(this->char_to_idx[j]);
      }

      this->sample(h_prev, c_prev, 100);
    }
  }
}

string LSTM::sample(Matrix h_prev, Matrix c_prev, size_t size) { return ""; }
