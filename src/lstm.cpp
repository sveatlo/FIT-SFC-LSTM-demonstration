#include "lstm.h"
#include "matrix.h"
#include <algorithm>
#include <chrono>
#include <float.h>
#include <iostream>
#include <map>
#include <math.h>
#include <random>
#include <sstream>
#include <string>

using namespace std;

LSTM::LSTM(map<char, size_t> _char_to_idx, map<size_t, char> _idx_to_char,
           size_t _vocab_size, size_t _n_h, size_t _seq_len, double _beta1,
           double _beta2)
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
  Matrix bi(this->n_h, 1, 0);
  this->params.insert(make_pair("Wi", wi * std));
  this->params.insert(make_pair("bi", bi));

  // cell state gate
  Matrix wc(this->n_h, this->n_h + this->vocab_size, 0);
  wc.randomize(-1, 1);
  Matrix bc(this->n_h, 1, 0);
  this->params.insert(make_pair("Wc", wc * std));
  this->params.insert(make_pair("bc", bc));

  // output gate
  Matrix wo(this->n_h, this->n_h + this->vocab_size, 0);
  wo.randomize(-1, 1);
  Matrix bo(this->n_h, 1, 0);
  this->params.insert(make_pair("Wo", wo * std));
  this->params.insert(make_pair("bo", bo));

  // output
  Matrix wv(this->vocab_size, this->n_h, 1);
  wv.randomize(-1, 1);
  Matrix bv(this->vocab_size, 1, 0);
  this->params.insert(make_pair("Wv", wv * (1 / sqrt(this->vocab_size))));
  this->params.insert(make_pair("bv", bv));

  for (auto const &item : this->params) {
    string param_name = item.first;
    Matrix param_matrix = item.second;

    this->grads["d" + param_name] =
        Matrix(param_matrix.rows_n(), param_matrix.cols_n(), 0);

    this->adam_params["m" + param_name] =
        Matrix(param_matrix.rows_n(), param_matrix.cols_n(), 0);
    this->adam_params["v" + param_name] =
        Matrix(param_matrix.rows_n(), param_matrix.cols_n(), 0);
  }

  this->smooth_loss = -1 * log(1.0f / this->vocab_size) * this->seq_len;

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  this->sample_random_generator = default_random_engine(seed);
}

Matrix LSTM::sigmoid(Matrix x) { return ((x * -1).exp() + 1).divides(1); }

Matrix LSTM::softmax(Matrix x) {
  Matrix e_x = (x - x.max()).exp();

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

void LSTM::update_params(double lr, size_t batch_n) {

  for (auto &item : this->params) {
    string key = item.first;

    Matrix tmp = this->grads["d" + key] * (1 - this->beta1);
    this->adam_params["m" + key] =
        this->adam_params["m" + key] * this->beta1 + tmp;

    tmp = this->grads["d" + key].pow(2);
    tmp = tmp * (1 - this->beta2);
    this->adam_params["v" + key] =
        this->adam_params["v" + key] * this->beta2 + tmp;

    Matrix m_correlated = this->adam_params["m" + key] /
                          (1 - pow(this->beta1, static_cast<double>(batch_n)));
    Matrix v_correlated = this->adam_params["v" + key] /
                          (1 - pow(this->beta2, static_cast<double>(batch_n)));

    Matrix tmp1 = (m_correlated * lr);
    Matrix tmp2 = (v_correlated.sqrt() + 1e-5);
    tmp = tmp1 / tmp2;
    this->params[key] = this->params[key] - tmp;
  }
}

LSTM_step_data LSTM::forward_step(Matrix x, Matrix h_prev, Matrix c_prev) {
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

LSTM_backward_return LSTM::backward_step(size_t y, Matrix y_hat, Matrix dh_next,
                                         Matrix dc_next, Matrix c_prev,
                                         Matrix z, Matrix f, Matrix i,
                                         Matrix c_hat, Matrix c, Matrix o,
                                         Matrix h) {

  Matrix dv = y_hat;
  dv(y, 0) -= 1;

  Matrix hT = h.transpose();
  Matrix tmp = dv.dot(hT);
  this->grads["dWv"] = this->grads["dWv"] + tmp;
  this->grads["dbv"] = this->grads["dbv"] + dv;

  Matrix wvT = this->params["Wv"].transpose();
  Matrix dh = wvT.dot(dv) + dh_next;

  Matrix c_tanh = c.tanh();
  Matrix do_ = dh * c_tanh;
  Matrix one_minus_o = (o * -1 + 1); // 1 - o
  Matrix da_o = do_ * o * one_minus_o;
  Matrix zT = z.transpose();
  tmp = da_o.dot(zT);
  this->grads["dWo"] = this->grads["dWo"] + tmp;
  this->grads["dbo"] = this->grads["dbo"] + da_o;

  tmp = c.tanh().pow(2);
  tmp = (tmp * -1) + 1;
  Matrix dc = dh * o * tmp;
  dc = dc + dc_next;

  Matrix dc_hat = dc * i;
  tmp = ((c_hat.pow(2)) * -1) + 1;
  Matrix da_c = dc_hat * tmp;
  tmp = da_c.dot(zT);
  this->grads["dWc"] = this->grads["dWc"] + tmp;
  this->grads["dbc"] = this->grads["dbc"] + da_c;

  Matrix di = dc * c_hat;
  tmp = (i * -1 + 1);
  Matrix da_i = di * i * tmp;
  tmp = da_i.dot(zT);
  this->grads["dWi"] = this->grads["dWi"] + tmp;
  this->grads["dbi"] = this->grads["dbi"] + da_i;

  Matrix df = dc * c_prev;
  tmp = (f * -1 + 1);
  Matrix da_f = df * f * tmp;
  tmp = da_f.dot(zT);
  this->grads["dWf"] = this->grads["dWf"] + tmp;
  this->grads["dbf"] = this->grads["dbf"] + da_f;

  Matrix dz = this->params["Wf"].transpose();
  dz = dz.dot(da_f);

  tmp = this->params["Wi"].transpose();
  tmp = tmp.dot(da_i);
  dz = dz + tmp;
  tmp = this->params["Wc"].transpose();
  tmp = tmp.dot(da_c);
  dz = dz + tmp;
  tmp = this->params["Wo"].transpose();
  tmp = tmp.dot(da_o);
  dz = dz + tmp;

  auto x = dz.ravel();
  x = vector<double>(x.begin(), x.begin() + this->n_h);

  Matrix dh_prev(x);
  dh_prev.reshape(this->n_h, 1);

  Matrix dc_prev = f * dc;

  return LSTM_backward_return{
      .dh_prev = dh_prev,
      .dc_prev = dc_prev,
  };
}

LSTM_forward_backward_return LSTM::forward_backward(vector<size_t> x_batch,
                                                    vector<size_t> y_batch,
                                                    Matrix h_prev,
                                                    Matrix c_prev) {
  map<size_t, Matrix> x, z;
  map<long int, Matrix> f, i, c, c_hat, o;
  map<long int, Matrix> y_hat, v, h;

  h[-1] = h_prev;
  c[-1] = c_prev;

  double loss = 0;
  for (size_t t = 0; t < this->seq_len; t++) {
    x[t] = Matrix(this->vocab_size, 1, 0);
    x[t](x_batch[t], 0) = 1;

    LSTM_step_data forward_res = this->forward_step(x[t], h[t - 1], c[t - 1]);

    y_hat[t] = forward_res.y_hat;
    v[t] = forward_res.v;
    h[t] = forward_res.h;
    o[t] = forward_res.o;
    c[t] = forward_res.c;
    c_hat[t] = forward_res.c_hat;
    i[t] = forward_res.i;
    f[t] = forward_res.f;
    z[t] = forward_res.z;

    loss += -1 * log(y_hat[t](y_batch[t], 0));
  }

  this->reset_grads();

  Matrix dh_next(h[0].rows_n(), h[0].cols_n(), 0);
  Matrix dc_next(c[0].rows_n(), c[0].cols_n(), 0);

  for (size_t t = this->seq_len - 1; t > 0; t--) {
    LSTM_backward_return backward_res =
        this->backward_step(y_batch[t], y_hat[t], dh_next, dc_next, c[t - 1],
                            z[t], f[t], i[t], c_hat[t], c[t], o[t], h[t]);
    dh_next = backward_res.dh_prev;
    dc_next = backward_res.dc_prev;
  }

  return LSTM_forward_backward_return{
      .loss = loss,
      .h = h[this->seq_len],
      .c = c[this->seq_len],
  };
}

LSTM_training_res LSTM::train(vector<char> _X, size_t epochs,
                              double lr = 0.001) {
  int num_batches = _X.size() / this->seq_len;
  vector<char> X(_X.begin(), _X.begin() + num_batches * this->seq_len);
  vector<double> losses;

  for (size_t epoch = 0; epoch < epochs; epoch++) {
    cout << "Starting epoch no." << epoch << " of " << X.size() / this->seq_len
         << " sequences" << endl;
    Matrix h_prev(this->n_h, 1, 0);
    Matrix c_prev(this->n_h, 1, 0);

    int delete_n = 0;
    for (size_t i = 0; i < X.size(); i += this->seq_len) {
      for (int d = 0; d < delete_n; d++) {
        cout << "\b";
      }

      cout << ".";

      stringstream ss;
      ss << "(loss = " << this->smooth_loss << ")";
      cout << ss.str();

      delete_n = ss.str().length();

      cout.flush();

      int batch_num = epoch * epochs + i / this->seq_len + 1;

      // prepare data
      vector<size_t> x_batch, y_batch;
      for (size_t j = i; j < i + this->seq_len; j++) {
        char c = X[j];
        x_batch.push_back(this->char_to_idx[c]);
      }
      for (size_t j = i + 1; j < i + this->seq_len + 1; j++) {
        char c = X[j];
        y_batch.push_back(this->char_to_idx[c]);
      }

      // forward-backward on batch
      LSTM_forward_backward_return batch_res =
          this->forward_backward(x_batch, y_batch, h_prev, c_prev);

	  this->smooth_loss = batch_res.loss;
	  // this->smooth_loss = this->smooth_loss * 0.999 + batch_res.loss * 0.001;
      losses.push_back(this->smooth_loss);

      this->clip_grads();

      this->update_params(lr, batch_num);
    }

    cout << endl;
    cout << "---------------Epoch " << epoch << "----------------------------"
         << endl;
    cout << "Loss: " << this->smooth_loss << endl;
    cout << "Sample: " << this->sample(h_prev, c_prev, 100);
    cout << endl;
    cout << "--------------------------------------------------" << endl;
  }

  // return make_pair(losses, this->params);
  return LSTM_training_res{
      .lossses = losses,
      .params = this->params,
  };
}

string LSTM::sample(Matrix h_prev, Matrix c_prev, size_t size) {
  Matrix x(this->vocab_size, 1, 0);
  Matrix h = h_prev;
  Matrix c = c_prev;

  string sample = "";
  for (size_t i = 0; i < size; i++) {
    LSTM_step_data res = this->forward_step(x, h, c);
    vector<double> probabilities = res.y_hat.ravel();
    h = res.h;
    c = res.c;

    std::discrete_distribution<int> distribution(probabilities.begin(),
                                                 probabilities.end());
    const size_t idx = distribution(this->sample_random_generator);

    sample += this->idx_to_char[idx];
  }

  return sample;
}
