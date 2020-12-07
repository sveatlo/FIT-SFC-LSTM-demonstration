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

LSTM::LSTM(map<char, unsigned> _char_to_idx, map<unsigned, char> _idx_to_char,
					 unsigned _vocab_size, unsigned _n_h, unsigned _seq_len, double _beta1,
					 double _beta2)
		: char_to_idx(_char_to_idx), idx_to_char(_idx_to_char),
			vocab_size(_vocab_size), n_h(_n_h), seq_len(_seq_len), beta1(_beta1),
			beta2(_beta2) {

	// Xavier initialization
	double sd = 1.0/ sqrt(this->vocab_size + this->n_h);

	// forget gate
	this->params.insert(make_pair("Wf", Param("Wf", Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size) * sd + 0.5)));
	// this->params.insert(make_pair("Wf", Param("Wf", Matrix<double>(this->n_h, this->n_h + this->vocab_size, 1) * sd + 0.5)));
	this->params.insert(make_pair("bf", Param("bf", Matrix<double>(this->n_h, 1, 0))));

	// input gate
	this->params.insert(make_pair("Wi", Param("Wi", Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size) * sd + 0.5)));
	// this->params.insert(make_pair("Wi", Param("Wi", Matrix<double>(this->n_h, this->n_h + this->vocab_size, 1) * sd + 0.5)));
	this->params.insert(make_pair("bi", Param("bi", Matrix<double>(this->n_h, 1, 0))));

	// cell gate
	this->params.insert(make_pair("Wc", Param("Wc", Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size) * sd)));
	// this->params.insert(make_pair("Wc", Param("Wc", Matrix<double>(this->n_h, this->n_h + this->vocab_size, 1) * sd)));
	this->params.insert(make_pair("bc", Param("bc", Matrix<double>(this->n_h, 1, 0))));

	// output gate
	this->params.insert(make_pair("Wo", Param("Wo", Matrix<double>::randn(this->n_h, this->n_h + this->vocab_size) * sd + 0.5)));
	// this->params.insert(make_pair("Wo", Param("Wo", Matrix<double>(this->n_h, this->n_h + this->vocab_size, 1) * sd + 0.5)));
	this->params.insert(make_pair("bo", Param("bo", Matrix<double>(this->n_h, 1, 0))));

	// final prediction layer
	this->params.insert(make_pair("Wv", Param("Wv", Matrix<double>::randn(this->vocab_size, this->n_h) * sd)));
	// this->params.insert(make_pair("Wv", Param("Wv", Matrix<double>(this->vocab_size, this->n_h, 1) * sd)));
	this->params.insert(make_pair("bv", Param("bv", Matrix<double>(this->vocab_size, 1, 0))));


	this->smooth_loss = -1 * log(1.0f / this->vocab_size) * this->seq_len;
    //
	// unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	// this->sample_random_generator = default_random_engine(seed);
}

Matrix<double> LSTM::sigmoid(Matrix<double> x) {
	// 1 / (1 + e^(-x))
	return ((x * -1).exp() + 1).divides(1);
}
Matrix<double> LSTM::dsigmoid(Matrix<double> x) {
	// x * (1 - x)
	auto tmp = ((x * -1) + 1);
	return x * tmp;
}
Matrix<double> LSTM::dtanh(Matrix<double> x) {
	// 1 - x*x
	return (x.pow(2) * -1) + 1;
}
Matrix<double> LSTM::softmax(Matrix<double> x) {
	// e^x / sum(x)
	Matrix<double> e_x = x.exp();
	return e_x / (e_x.sum() + 1e-8);
}

void LSTM::clip_grads() {
	for (auto &item : this->params) {
		this->params[item.first].d.clip(-5, 5);
	}
}

void LSTM::reset_grads() {
	for (auto &item : this->params) {
		this->params[item.first].d = Matrix<double>(item.second.d.rows_n(), item.second.d.cols_n(), 0);
	}
}

void LSTM::update_params(double lr) {
	for (auto &p : this->params) {
		p.second.m += p.second.d * p.second.d;

		Matrix<double> tmp = (p.second.m + 1e-8).sqrt();
		p.second.v += ((p.second.d * lr) / tmp) * -1;
	}
}

LSTM_step_data LSTM::forward_step(Matrix<double> x, Matrix<double> h_prev, Matrix<double> c_prev) {
    assert(x.rows_n() == this->vocab_size && x.cols_n() == 1);
    assert(h_prev.rows_n() == this->n_h && h_prev.cols_n() == 1);
    assert(c_prev.rows_n() == this->n_h && c_prev.cols_n() == 1);

	Matrix<double> tmp;
	Matrix<double>& Wf = this->params["Wf"].v;
	Matrix<double>& bf = this->params["bf"].v;
	Matrix<double>& Wi = this->params["Wi"].v;
	Matrix<double>& bi = this->params["bi"].v;
	Matrix<double>& Wc = this->params["Wc"].v;
	Matrix<double>& bc = this->params["bc"].v;
	Matrix<double>& Wo = this->params["Wo"].v;
	Matrix<double>& bo = this->params["bo"].v;
	Matrix<double>& Wv = this->params["Wv"].v;
	Matrix<double>& bv = this->params["bv"].v;


	Matrix<double> z = h_prev.vstack(x);

	Matrix<double> f = this->sigmoid(Wf.dot(z) + bf);
	Matrix<double> i = this->sigmoid(Wi.dot(z) + bi);
	Matrix<double> c_bar = (Wc.dot(z) + bc).tanh();
	Matrix<double> o = this->sigmoid(Wo.dot(z) + bo);

	tmp =  i * c_bar;
	Matrix<double> c = f * c_prev + tmp;

	tmp = c.tanh();
	Matrix<double> h = o * tmp;
	Matrix<double> v = Wv.dot(h) + bv;
	Matrix<double> y = this->softmax(v);

	LSTM_step_data step_data = {
			.y = y,
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

LSTM_backward_return LSTM::backward_step(
		unsigned target_idx,
		Matrix<double> dh_next,
		Matrix<double> dc_next,
		Matrix<double> c_prev,
		Matrix<double> z,
		Matrix<double> f,
		Matrix<double> i,
		Matrix<double> c_bar,
		Matrix<double> c,
		Matrix<double> o,
		Matrix<double> h,
		Matrix<double> v,
		Matrix<double> y) {
    assert(z.rows_n() == this->vocab_size + this->n_h && z.cols_n() == 1);
    assert(v.rows_n() == this->vocab_size && v.cols_n() == 1);
    assert(y.rows_n() == this->vocab_size && y.cols_n() == 1);
	assert(dh_next.rows_n() == this->n_h && dh_next.cols_n() == 1);
	assert(dc_next.rows_n() == this->n_h && dc_next.cols_n() == 1);
	assert(c_prev.rows_n() == this->n_h && c_prev.cols_n() == 1);
	assert(f.rows_n() == this->n_h && f.cols_n() == 1);
	assert(i.rows_n() == this->n_h && i.cols_n() == 1);
	assert(o.rows_n() == this->n_h && o.cols_n() == 1);
	assert(h.rows_n() == this->n_h && h.cols_n() == 1);
	assert(c_bar.rows_n() == this->n_h && c_bar.cols_n() == 1);

	Matrix<double> tmp;
	Matrix<double>& Wf = this->params["Wf"].v;
	Matrix<double>& Wi = this->params["Wi"].v;
	Matrix<double>& Wc = this->params["Wc"].v;
	Matrix<double>& Wo = this->params["Wo"].v;
	Matrix<double>& Wv = this->params["Wv"].v;


	Matrix<double> dv(y);
	dv(target_idx, 0) -= 1;

	Matrix<double> zT = z.transpose();
	Matrix<double> hT = h.transpose();


	this->params["Wv"].d += dv.dot(hT);
	this->params["bv"].d += dv;

	Matrix<double> dh = Wv.transpose().dot(dv);
	dh += dh_next;
	Matrix<double> c_tanh = c.tanh();
	Matrix<double> do_ = dh * c_tanh;
	do_ = this->dsigmoid(o) * do_;  // gets fucking big (6.5838617773635592e+187)
	this->params["Wo"].d += do_.dot(zT);
	this->params["bo"].d += do_;

	Matrix<double> dc(dc_next);
	tmp = this->dtanh(c_tanh);
	dc += dh * o * tmp;
	Matrix<double> dc_bar = dc * i;
	dc_bar = this->dtanh(c_bar) * dc_bar; // gets really fucking small (-2.6288897253566143e+42)
	this->params["Wc"].d += dc_bar.dot(zT);
	this->params["bc"].d += dc_bar;

	Matrix<double> di = dc * c_bar;
	di = this->dsigmoid(i) * di;
	this->params["Wi"].d += di.dot(zT);
	this->params["bi"].d += di;

	Matrix<double> df = dc * c_prev;
	df = this->dsigmoid(f) * df;
	this->params["Wf"].d += df.dot(zT);
	this->params["bf"].d += df;

	Matrix<double> dz = Wf.transpose().dot(df);
	dz += Wi.transpose().dot(di);
	dz += Wc.transpose().dot(dc_bar);
	dz += Wo.transpose().dot(do_);

	LSTM_backward_return r;

	vector<double> dh_prev_data = dz.ravel();
	dh_prev_data =
	  vector<double>(dh_prev_data.begin(),
					 dh_prev_data.begin() + this->n_h * dz.cols_n());
	r.dh_prev = Matrix<double>(dh_prev_data);
	r.dh_prev.reshape(this->n_h, 1);

	r.dc_prev = f * dc;

	return r;
}

LSTM_forward_backward_return LSTM::forward_backward(vector<unsigned> x_batch, vector<unsigned> y_batch, Matrix<double> h_prev, Matrix<double> c_prev) {
	vector<LSTM_step_data> progress;
	LSTM_step_data init;
	init.h = h_prev;
	init.c = c_prev;
	progress.push_back(init);

	double loss = 0;
	for (unsigned t = 0; t < this->seq_len; t++) {
		Matrix<double> x(this->vocab_size, 1, 0);
		x(x_batch[t], 0) = 1;

		LSTM_step_data forward_res = this->forward_step(x, progress.back().h, progress.back().c);

		progress.push_back(forward_res);

		loss += -1 * log(forward_res.y(y_batch[t], 0));
	}

	this->reset_grads();

	Matrix<double> dh_next(h_prev.rows_n(), h_prev.cols_n(), 0);
	Matrix<double> dc_next(c_prev.rows_n(), c_prev.cols_n(), 0);

	for (unsigned t = this->seq_len; t > 0; t--) { // forward pass ended at index this->seq_len because it started at 1, not 0
		// cout << "backward step @ time t == " << t << endl;
		LSTM_backward_return backward_res = this->backward_step(
				y_batch[t - 1], // chars in batch start from 0
				dh_next,
				dc_next,
				progress.at(t).c,
				progress.at(t).z,
				progress.at(t).f,
				progress.at(t).i,
				progress.at(t).c_bar,
				progress.at(t).c,
				progress.at(t).o,
				progress.at(t).h,
				progress.at(t).v,
				progress.at(t).y
		);

		dh_next = backward_res.dh_prev;
		dc_next = backward_res.dc_prev;
	}
	this->clip_grads();

	return LSTM_forward_backward_return{
			.loss = loss,
			.h = progress.back().h,
			.c = progress.back().c,
	};
}

LSTM_training_res LSTM::train(vector<char> _X, unsigned epochs, double lr = 0.001) {
	int num_batches = _X.size() / this->seq_len;
	vector<char> X(_X.begin(), _X.begin() + num_batches * this->seq_len);
	vector<double> losses;

	for (unsigned epoch = 0; epoch < epochs; epoch++) {
		cout << "Starting epoch no." << epoch << " with " << X.size() / this->seq_len
				 << " batches" << endl;
		Matrix<double> h_prev(this->n_h, 1, 0);
		Matrix<double> c_prev(this->n_h, 1, 0);

		// int delete_n = 0;
		for (unsigned i = 0; i < X.size() - this->seq_len; i += this->seq_len) {
			int batch_num = epoch * epochs + i / this->seq_len;
			cout << "\rEpoch " << epoch << ": batch " << batch_num << "/" << X.size() / this->seq_len << " (loss: " << this->smooth_loss << ")";
			cout.flush();


			// prepare data
			vector<unsigned> x_batch, y_batch;
			for (unsigned j = i; j < i + this->seq_len; j++) {
				char c = X[j];
				x_batch.push_back(this->char_to_idx[c]);
			}
			for (unsigned j = i + 1; j < i + this->seq_len + 1; j++) {
				char c = X[j];
				y_batch.push_back(this->char_to_idx[c]);
			}

			// forward-backward on batch
			LSTM_forward_backward_return batch_res = this->forward_backward(x_batch, y_batch, h_prev, c_prev);

			// this->smooth_loss = batch_res.loss;
			this->smooth_loss = this->smooth_loss * 0.99 + batch_res.loss * 0.01;
			losses.push_back(this->smooth_loss);


			this->update_params(lr);
		}

		cout << endl;
		cout << "---------------Epoch " << epoch << "----------------------------"
				 << endl;
		cout << "Loss: " << this->smooth_loss << endl;
		cout << "Sample: " << this->sample(100, 't');
		cout << endl;
		cout << "--------------------------------------------------" << endl;
	}

	// return make_pair(losses, this->params);
	return LSTM_training_res{
			.lossses = losses,
			.params = this->params,
	};
}

string LSTM::sample(unsigned size, char seed) {
	Matrix<double> x(this->vocab_size, 1, 0);
	if (seed != '\0') {
		x(this->char_to_idx[seed], 0) = 1;
	}

	Matrix<double> h(this->n_h, 1, 0);
	Matrix<double> c(this->n_h, 1, 0);

	string sample = "";
	for (unsigned i = 0; i < size; i++) {
		LSTM_step_data res = this->forward_step(x, h, c);

		vector<double> probabilities = res.y.ravel();
		h = res.h;
		c = res.c;

		std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
		const unsigned idx = distribution(this->sample_random_generator);

		x.fill(0);
		x(idx, 0) = 1;

		sample += this->idx_to_char[idx];
	}

	return sample;
}
