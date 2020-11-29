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
  Matrix wf = Matrix::randn(this->n_h, this->n_h + this->vocab_size);
  Matrix bf = Matrix(this->n_h, 1, 1);
  this->params.insert(make_pair("Wf", wf * std));
  this->params.insert(make_pair("bf", bf));

  // input gate
  Matrix wi = Matrix::randn(this->n_h, this->n_h + this->vocab_size);
  Matrix bi(this->n_h, 1, 0);
  this->params.insert(make_pair("Wi", wi * std));
  this->params.insert(make_pair("bi", bi));

  // cell state gate
  Matrix wc = Matrix::randn(this->n_h, this->n_h + this->vocab_size);
  Matrix bc(this->n_h, 1, 0);
  this->params.insert(make_pair("Wc", wc * std));
  this->params.insert(make_pair("bc", bc));

  // output gate
  Matrix wo = Matrix::randn(this->n_h, this->n_h + this->vocab_size);
  Matrix bo(this->n_h, 1, 0);
  this->params.insert(make_pair("Wo", wo * std));
  this->params.insert(make_pair("bo", bo));

  // output
  Matrix wy(this->vocab_size, this->n_h, 1);
  Matrix by(this->vocab_size, 1, 0);
  this->params.insert(make_pair("Wy", wy * (1 / sqrt(this->vocab_size))));
  this->params.insert(make_pair("by", by));

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

Matrix LSTM::one_hot_encode(size_t c) {
  Matrix m(this->vocab_size, 1, 0);
  m(c, 0) = 1;
  return m;
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


void LSTM::update_params(double lr) {
  size_t batch_n = 100;
  for (auto &item : this->params) {
    string key = item.first;

        Matrix tmp = this->grads["d" + key] * (1 - this->beta1);
        this->adam_params["m" + key] = this->adam_params["m" + key] * this->beta1 + tmp;

        tmp = this->grads["d" + key].pow(2);
        tmp = tmp * (1 - this->beta2);
        this->adam_params["v" + key] = this->adam_params["v" + key] * this->beta2 + tmp;

        Matrix m_correlated = this->adam_params["m" + key] / (1 - pow(this->beta1, static_cast<double>(batch_n)));
        Matrix v_correlated = this->adam_params["v" + key] / (1 - pow(this->beta2, static_cast<double>(batch_n)));

        Matrix tmp1 = (m_correlated * lr);
        Matrix tmp2 = (v_correlated.sqrt() + 1e-5);
        tmp = tmp1 / tmp2;
        this->params[key] = this->params[key] - tmp;
  }
}
// void LSTM::update_params(double lr) {
//   for (auto &x : this->params) {
//     string key = x.first;
//
//     Matrix tmp;
//     Matrix d = this->grads["d" + key];
//     Matrix m_param = this->adam_params["m" + key];
//     Matrix v_param = this->adam_params["m" + key];
//
//     tmp = d.pow(2) * (1 - this->beta2);
//     m_param = m_param * this->beta2 + tmp;
//     this->adam_params["m" + key] = m_param;
//
//     tmp = d * (1 - this->beta1);
//     v_param = v_param * this->beta1 + tmp;
//     this->adam_params["v" + key] = v_param;
//
//     // param = param - learning_rate*((vparam)/(np.sqrt(m_param) + 1e-6))
//     tmp = m_param.sqrt() + 1e-6;
//     tmp = (v_param / tmp) * lr;
//     this->params[key] = this->params[key] - tmp;
//   }
// }
//
LSTM_cell_data LSTM::cell_forward(Matrix x, Matrix h_prev, Matrix c_prev) {
  Matrix tmp;

  // concatenated dataset [h_prev, x]
  Matrix z = h_prev.vstack(x);

  Matrix f = this->sigmoid(this->params["Wf"].dot(z) + this->params["bf"]);
  Matrix i = this->sigmoid(this->params["Wi"].dot(z) + this->params["bi"]);
  Matrix c_bar = (this->params["Wc"].dot(z) + this->params["bc"]).tanh();
  Matrix o = this->sigmoid(this->params["Wo"].dot(z) + this->params["bo"]);

  tmp = i * c_bar;
  Matrix c = f * c_prev + tmp;
  tmp = c.tanh();
  Matrix h = o * tmp;

  // Matrix y = this->sigmoid(this->params["Wy"].dot(h) + this->params["by"]);
  Matrix y = this->params["Wy"].dot(h) + this->params["by"];
  Matrix y_hat = this->softmax(y);

  LSTM_cell_data step_data = {
      .y_hat = y_hat,
      .y = y,
      .h = h,
      .o = o,
      .c = c,
      .c_bar = c_bar,
      .i = i,
      .f = f,
      .z = z,
      .c_prev = c_prev,
      .h_prev = h_prev,
  };

  return step_data;
}

LSTM_backward_return LSTM::cell_backward(Matrix dh_next, Matrix dc_next, size_t char_idx, LSTM_cell_data cd) {
	LSTM_backward_return r;
	Matrix tmp;
	Matrix hT = cd.h.transpose();
	Matrix zT = cd.z.transpose();

	Matrix dy = cd.y_hat;
	dy(char_idx, 0) -= 1;
	r.dWy = dy.dot(hT);
	r.dby = dy;

	Matrix dh = this->params["Wy"].transpose().dot(dy) + dh_next;
	tmp = cd.c.tanh();
	Matrix do_ = dh * tmp;

	tmp = (cd.o * -1 + 1);
    Matrix da_o = do_ * cd.o * tmp;
    r.dWo = da_o.dot(zT);
    r.dbo = da_o;

	tmp = (cd.c.tanh() * -1 + 1).pow(2);
	Matrix dc = dh * cd.o * tmp + dc_next;

	Matrix dc_bar = dc * cd.i;
	tmp = (cd.c_bar.pow(2) * -1 + 1);
	Matrix da_c = dc_bar * tmp;
	r.dWc = da_c.dot(zT);
	r.dbc = da_c;

	Matrix di = dc * cd.c_bar;
	tmp = (cd.i * -1 + 1);
	Matrix da_i = di * cd.i * tmp;
	r.dWi = da_i.dot(zT);
	r.dbi = da_i;

	Matrix df = dc * cd.c_prev;
	tmp = (cd.f * -1 + 1);
	Matrix da_f = df * cd.f * tmp;
	r.dWf = da_f.dot(zT);
	r.dbf = da_f;


	tmp = this->params["Wf"].transpose().dot(da_f);
	Matrix dh_prev_full = tmp;
	tmp = this->params["Wi"].transpose().dot(da_i);
	dh_prev_full = dh_prev_full + tmp;
	tmp = this->params["Wc"].transpose().dot(da_c);
	dh_prev_full = dh_prev_full + tmp;
	tmp = this->params["Wo"].transpose().dot(da_o);
	dh_prev_full = dh_prev_full + tmp;

	vector<double> dh_prev_data = dh_prev_full.ravel();
	dh_prev_data =
	  vector<double>(dh_prev_data.begin(),
					 dh_prev_data.begin() + this->n_h * dh_prev_full.cols_n());
	r.dh_prev = Matrix(dh_prev_data);
	r.dh_prev.reshape(this->n_h, 1);

	r.dc_prev = cd.f * dc;

	return r;
}

LSTM_optimization_res LSTM::optimize(vector<size_t> x_batch,
                                     vector<size_t> y_batch, Matrix h_prev,
                                     Matrix c_prev, double lr) {
  vector<LSTM_cell_data> progress;
  LSTM_cell_data init;
  init.h = h_prev, init.c = c_prev, progress.push_back(init);

  double loss = 0;
  // run forward on all steps
  for (size_t t = 1; t < this->seq_len + 1; t++) { // index 0 = init, starting from 1
    Matrix x_t = this->one_hot_encode(x_batch[t-1]);
    Matrix y_t = this->one_hot_encode(y_batch[t-1]);

    LSTM_cell_data res =
        this->cell_forward(x_t, progress.back().h, progress.back().c);
    progress.push_back(res);

    // loss += (y_t - res.y_hat).pow(2).sum();
	loss += -1 * log(res.y_hat(y_batch[t-1], 0));
  }

  // run backward accumulating gradients
  this->reset_grads();
  Matrix dh_next(progress.front().h.rows_n(), progress.front().h.cols_n(), 0);
  Matrix dc_next(progress.front().c.rows_n(), progress.front().c.cols_n(), 0);
  for (size_t t = this->seq_len; t > 0; t--) { // forward pass ended at index this->seq_len because it started at 1, not 0
    LSTM_backward_return backward_res =
        this->cell_backward(dh_next, dc_next, y_batch[t-1], progress.at(t));

    dh_next = backward_res.dh_prev;
    dc_next = backward_res.dc_prev;

    this->grads["dWf"] = this->grads["dWf"] + backward_res.dWf;
    this->grads["dWi"] = this->grads["dWi"] + backward_res.dWi;
    this->grads["dWc"] = this->grads["dWc"] + backward_res.dWc;
    this->grads["dWo"] = this->grads["dWo"] + backward_res.dWo;
    this->grads["dbf"] = this->grads["dbf"] + backward_res.dbf;
    this->grads["dbi"] = this->grads["dbi"] + backward_res.dbi;
    this->grads["dbc"] = this->grads["dbc"] + backward_res.dbc;
    this->grads["dbo"] = this->grads["dbo"] + backward_res.dbo;
  }

  // update
  // this->clip_grads();
  this->update_params(lr);

  return LSTM_optimization_res{
      .loss = loss,
      .h = progress.back().h,
      .c = progress.back().c,
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

      // int batch_num = epoch * epochs + i / this->seq_len + 1;

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
      LSTM_optimization_res batch_res =
          this->optimize(x_batch, y_batch, h_prev, c_prev, lr);

      // this->smooth_loss = this->smooth_loss * 0.999 + batch_res.loss * 0.001;
      this->smooth_loss = batch_res.loss;
      losses.push_back(this->smooth_loss);
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
    LSTM_cell_data res = this->cell_forward(x, h, c);
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
