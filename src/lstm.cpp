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
	double std = 1.0/ sqrt(this->vocab_size + this->n_h);

	// forget gate
	Matrix<double> wf = Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size);
	Matrix<double> bf = Matrix<double>(this->n_h, 1, 0);
	this->params.insert(make_pair("Wf", wf * std));
	this->params.insert(make_pair("bf", bf));

	// input gate
	Matrix<double> wi = Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size);
	Matrix<double> bi(this->n_h, 1, 0);
	this->params.insert(make_pair("Wi", wi * std));
	this->params.insert(make_pair("bi", bi));

	// cell state gate
	Matrix<double> wc = Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size);
	Matrix<double> bc(this->n_h, 1, 0);
	this->params.insert(make_pair("Wc", wc * std));
	this->params.insert(make_pair("bc", bc));

	// output gate
	Matrix<double> wo = Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size);
	Matrix<double> bo(this->n_h, 1, 0);
	this->params.insert(make_pair("Wo", wo * std));
	this->params.insert(make_pair("bo", bo));

	// output
	Matrix<double> wy(this->vocab_size, this->n_h, 1);
	Matrix<double> by(this->vocab_size, 1, 0);
	this->params.insert(make_pair("Wy", wy * (1 / sqrt(this->vocab_size))));
	this->params.insert(make_pair("by", by));

	for (auto const &item : this->params) {
		string param_name = item.first;
		Matrix<double> param_matrix = item.second;

		this->grads["d" + param_name] =
				Matrix<double>(param_matrix.rows_n(), param_matrix.cols_n(), 0);

		this->adam_params["m" + param_name] =
				Matrix<double>(param_matrix.rows_n(), param_matrix.cols_n(), 0);
		this->adam_params["v" + param_name] =
				Matrix<double>(param_matrix.rows_n(), param_matrix.cols_n(), 0);
	}

	this->smooth_loss = -1 * log(1.0f / this->vocab_size) * this->seq_len;

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	this->sample_random_generator = default_random_engine(seed);
}

Matrix<double> LSTM::sigmoid(Matrix<double> x) { return ((x * -1).exp() + 1).divides(1); }

Matrix<double> LSTM::softmax(Matrix<double> x) {
	Matrix<double> e_x = (x - x.max()).exp();

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
	// adam optmizer
	for (auto &item : this->params) {
		string key = item.first;

		Matrix<double> tmp = this->grads["d" + key] * (1 - this->beta1);
		this->adam_params["m" + key] = this->adam_params["m" + key] * this->beta1 + tmp;

		tmp = this->grads["d" + key].pow(2);
		tmp = tmp * (1 - this->beta2);
		this->adam_params["v" + key] = this->adam_params["v" + key] * this->beta2 + tmp;

		Matrix<double> m_correlated = this->adam_params["m" + key] / (1 - pow(this->beta1, static_cast<double>(batch_n)));
		Matrix<double> v_correlated = this->adam_params["v" + key] / (1 - pow(this->beta2, static_cast<double>(batch_n)));

		tmp = (v_correlated.sqrt() + 1e-8);
		this->params[key] -= (m_correlated * lr) / tmp;

		// cout << key << ": ";
		// this->params[key].print();
	}
}

LSTM_step_data LSTM::forward_step(Matrix<double> x, Matrix<double> h_prev, Matrix<double> c_prev) {
	Matrix<double> z = h_prev.vstack(x);

	Matrix<double> f = this->sigmoid(this->params["Wf"].dot(z) + this->params["bf"]);
	Matrix<double> i = this->sigmoid(this->params["Wi"].dot(z) + this->params["bi"]);
	Matrix<double> c_bar = (this->params["Wc"].dot(z) + this->params["bc"]).tanh();
	Matrix<double> o = this->sigmoid(this->params["Wo"].dot(z) + this->params["bo"]);

	Matrix<double> ctmp = i * c_bar;
	Matrix<double> c = f * c_prev + ctmp;
	Matrix<double> c_tanh = c.tanh();
	Matrix<double> h = o * c_tanh;
	Matrix<double> v = this->sigmoid(this->params["Wy"].dot(h) + this->params["by"]);
	Matrix<double> y_hat = this->softmax(v);

	LSTM_step_data step_data = {
			.y_hat = y_hat,
			.v = v,
			.h = h,
			.o = o,
			.c = c,
			.c_bar = c_bar,
			.i = i,
			.f = f,
			.z = z,
	};

	return step_data;
}

LSTM_backward_return LSTM::backward_step(size_t y, Matrix<double> y_hat, Matrix<double> dh_next,
																				 Matrix<double> dc_next, Matrix<double> c_prev,
																				 Matrix<double> z, Matrix<double> f, Matrix<double> i,
																				 Matrix<double> c_bar, Matrix<double> c, Matrix<double> o,
																				 Matrix<double> h) {

	Matrix<double> tmp;
	Matrix<double> dy = y_hat;
	dy(y, 0) -= 1;

	Matrix<double> hT = h.transpose();
	this->grads["dWy"] += dy.dot(hT);
	this->grads["dby"] += dy;

	Matrix<double> wyT = this->params["Wy"].transpose();
	Matrix<double> dh = wyT.dot(dy) + dh_next;

	Matrix<double> c_tanh = c.tanh();
	Matrix<double> do_ = dh * c_tanh;
	Matrix<double> one_minus_o = (o * -1 + 1); // 1 - o
	Matrix<double> da_o = do_ * o * one_minus_o;
	Matrix<double> zT = z.transpose();
	tmp = da_o.dot(zT);
	this->grads["dWo"] += tmp;
	this->grads["dbo"] += da_o;

	tmp = c.tanh().pow(2);
	tmp = (tmp * -1) + 1;
	Matrix<double> dc = dh * o * tmp;
	dc = dc + dc_next;

	Matrix<double> dc_bar = dc * i;
	tmp = ((c_bar.pow(2)) * -1) + 1;
	Matrix<double> da_c = dc_bar * tmp;
	tmp = da_c.dot(zT);
	this->grads["dWc"] += tmp;
	this->grads["dbc"] += da_c;

	Matrix<double> di = dc * c_bar;
	tmp = (i * -1 + 1);
	Matrix<double> da_i = di * i * tmp;
	tmp = da_i.dot(zT);
	this->grads["dWi"] += tmp;
	this->grads["dbi"] +=  da_i;

	Matrix<double> df = dc * c_prev;
	tmp = (f * -1 + 1);
	Matrix<double> da_f = df * f * tmp;
	tmp = da_f.dot(zT);
	this->grads["dWf"] += tmp;
	this->grads["dbf"] += da_f;

	Matrix<double> dz = this->params["Wf"].transpose().dot(da_f);
	tmp = this->params["Wi"].transpose().dot(da_i);
	dz = dz + tmp;
	tmp = this->params["Wc"].transpose().dot(da_c);
	dz = dz + tmp;
	tmp = this->params["Wo"].transpose().dot(da_o);
	dz = dz + tmp;

	auto x = dz.ravel();
	x = vector<double>(x.begin(), x.begin() + this->n_h * dz.cols_n());

	Matrix<double> dh_prev(x);
	dh_prev.reshape(this->n_h, 1);

	Matrix<double> dc_prev = f * dc;

	return LSTM_backward_return{
			.dh_prev = dh_prev,
			.dc_prev = dc_prev,
	};
}

LSTM_forward_backward_return LSTM::forward_backward(vector<size_t> x_batch, vector<size_t> y_batch, Matrix<double> h_prev, Matrix<double> c_prev) {
	map<size_t, Matrix<double>> x, z;
	map<long int, Matrix<double>> f, i, c, c_bar, o;
	map<long int, Matrix<double>> y_hat, v, h;

	h[-1] = h_prev;
	c[-1] = c_prev;

	double loss = 0;
	for (size_t t = 0; t < this->seq_len; t++) {
		x[t] = Matrix<double>(this->vocab_size, 1, 0);
		x[t](x_batch[t], 0) = 1;

		LSTM_step_data forward_res = this->forward_step(x[t], h[t - 1], c[t - 1]);

		y_hat[t] = forward_res.y_hat;
		v[t] = forward_res.v;
		h[t] = forward_res.h;
		o[t] = forward_res.o;
		c[t] = forward_res.c;
		c_bar[t] = forward_res.c_bar;
		i[t] = forward_res.i;
		f[t] = forward_res.f;
		z[t] = forward_res.z;

		loss += -1 * log(y_hat[t](y_batch[t], 0));
	}

	this->reset_grads();

	Matrix<double> dh_next(h[0].rows_n(), h[0].cols_n(), 0);
	Matrix<double> dc_next(c[0].rows_n(), c[0].cols_n(), 0);

	for (size_t t = this->seq_len - 1; t > 0; t--) {
		LSTM_backward_return backward_res =
				this->backward_step(y_batch[t], y_hat[t], dh_next, dc_next, c[t - 1], z[t], f[t], i[t], c_bar[t], c[t], o[t], h[t]);
		dh_next = backward_res.dh_prev;
		dc_next = backward_res.dc_prev;
	}

	return LSTM_forward_backward_return{
			.loss = loss,
			.h = h[this->seq_len],
			.c = c[this->seq_len],
	};
}

LSTM_training_res LSTM::train(vector<char> _X, size_t epochs, double lr = 0.001) {
	int num_batches = _X.size() / this->seq_len;
	vector<char> X(_X.begin(), _X.begin() + num_batches * this->seq_len);
	vector<double> losses;

	for (size_t epoch = 0; epoch < epochs; epoch++) {
		cout << "Starting epoch no." << epoch << " with " << X.size() / this->seq_len
				 << " sequences" << endl;
		Matrix<double> h_prev(this->n_h, 1, 0);
		Matrix<double> c_prev(this->n_h, 1, 0);

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
			LSTM_forward_backward_return batch_res = this->forward_backward(x_batch, y_batch, h_prev, c_prev);

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

string LSTM::sample(Matrix<double> h_prev, Matrix<double> c_prev, size_t size) {
	Matrix<double> x(this->vocab_size, 1, 0);
	Matrix<double> h = h_prev;
	Matrix<double> c = c_prev;

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
