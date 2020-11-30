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

LSTM::LSTM(map<char, size_t> char_to_idx, map<size_t, char> idx_to_char, size_t vocab_size, size_t n_h, size_t seq_len) {
	this->char_to_idx = char_to_idx;
	this->idx_to_char = idx_to_char;
	this->vocab_size = vocab_size;
	this->n_h = n_h;
	this->seq_len = seq_len;


	this->params["We"] = Matrix::randn(this->n_h, this->n_h);
	this->params["Wf"] = Matrix::randn(this->n_h, this->n_h);
	this->params["Wg"] = Matrix::randn(this->n_h, this->n_h);
	this->params["Wq"] = Matrix::randn(this->n_h, this->n_h);

	this->params["Ue"] = Matrix::randn(this->n_h, this->vocab_size);
	this->params["Uf"] = Matrix::randn(this->n_h, this->vocab_size);
	this->params["Ug"] = Matrix::randn(this->n_h, this->vocab_size);
	this->params["Uq"] = Matrix::randn(this->n_h, this->vocab_size);

	this->params["be"] = Matrix(1, this->n_h, 0);
	this->params["bf"] = Matrix(1, this->n_h, 0);
	this->params["bg"] = Matrix(1, this->n_h, 0);
	this->params["bq"] = Matrix(1, this->n_h, 0);

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

vector<double> LSTM::one_hot_encode(size_t c) {
  vector<double> encoding(this->vocab_size, 0);
  encoding[c] = 1;
  return encoding;
}

//
// cell_forward computes h_next and c_next for current input
// it also returns all other values for use in BPTT
//
LSTM_cell_fwd_res LSTM::cell_forward(Matrix x_t, Matrix h_prev, Matrix c_prev) {
	Matrix tmp1, tmp2;

	Matrix& We = this->params["We"];
	Matrix& Wf = this->params["Wf"];
	Matrix& Wg = this->params["Wg"];
	Matrix& Wq = this->params["Wq"];
	Matrix& Ue = this->params["Ue"];
	Matrix& Uf = this->params["Uf"];
	Matrix& Ug = this->params["Ug"];
	Matrix& Uq = this->params["Uq"];
	Matrix& be = this->params["be"];
	Matrix& bf = this->params["bf"];
	Matrix& bg = this->params["bg"];
	Matrix& bq = this->params["bq"];

	LSTM_cell_fwd_res res;
	res.x_t = x_t;
	res.h_prev = h_prev;
	res.c_prev = c_prev;

	tmp1 = Ue.transpose();
	tmp1 = x_t.dot(tmp1);
	tmp2 = We.transpose();
	tmp2 = h_prev.dot(tmp2);
	res.e_t = this->sigmoid(be + tmp1 + tmp2);

	tmp1 = Uf.transpose();
	tmp1 = x_t.dot(tmp1);
	tmp2 = Wf.transpose();
	tmp2 = h_prev.dot(tmp2);
	res.f_t = this->sigmoid(bf + tmp1 + tmp2);

	tmp1 = Ug.transpose();
	tmp1 = x_t.dot(tmp1);
	tmp2 = Wg.transpose();
	tmp2 = h_prev.dot(tmp2);
	res.g_t = this->sigmoid(bg + tmp1 + tmp2);

	tmp1 = Uq.transpose();
	tmp1 = x_t.dot(tmp1);
	tmp2 = Wq.transpose();
	tmp2 = h_prev.dot(tmp2);
	res.q_t = this->sigmoid(bq + tmp1 + tmp2);

	tmp1 = res.g_t * res.e_t;
	res.c_next = res.f_t * c_prev + tmp1;
	tmp1 = res.c_next.tanh();
    res.h_next = res.q_t * tmp1;

	return res;
}

LSTM_cell_bwd_res LSTM::cell_backward(Matrix dh_next, Matrix dc_next, LSTM_cell_fwd_res fwd_res) {
	LSTM_cell_bwd_res res;
	Matrix tmp;

	Matrix& We = this->params["We"];
	Matrix& Wf = this->params["Wf"];
	Matrix& Wg = this->params["Wg"];
	Matrix& Wq = this->params["Wq"];
	// Matrix& Ue = this->params["Ue"];
	// Matrix& Uf = this->params["Uf"];
	// Matrix& Ug = this->params["Ug"];
	// Matrix& Uq = this->params["Uq"];
	// Matrix& be = this->params["be"];
	// Matrix& bf = this->params["bf"];
	// Matrix& bg = this->params["bg"];
	// Matrix& bq = this->params["bq"];

	Matrix& c_prev = fwd_res.c_prev;
	Matrix& c_next = fwd_res.c_next;
	Matrix& x_t = fwd_res.x_t;
	Matrix& e_t = fwd_res.e_t;
	Matrix& f_t = fwd_res.f_t;
	Matrix& g_t = fwd_res.g_t;
	Matrix& q_t = fwd_res.q_t;
	Matrix& h_prev = fwd_res.h_prev;

	Matrix c_tan = c_next.tanh();

	tmp = (c_tan.pow(2) * -1 + 1);
	dc_next = dh_next * q_t * tmp + dc_next;

	// forget gate
	Matrix df_step = dc_next * c_prev;
	tmp = (f_t * -1 + 1);
	Matrix dsigmoid_f = f_t * tmp;
	Matrix f_tmp = df_step * dsigmoid_f;
	Matrix f_tmpT = f_tmp.transpose();
	Matrix dUf_step = f_tmpT.dot(x_t);
	Matrix dWf_step = f_tmpT.dot(h_prev);
	Matrix dbf_step = f_tmp.sum_cols();

	// input gate
	Matrix dg_step = dc_next * e_t;
	tmp = (g_t * -1 + 1);
	Matrix dsigmoid_g = g_t * tmp;
	Matrix g_tmp = dg_step * dsigmoid_g;
	Matrix g_tmpT = g_tmp.transpose();
	Matrix dUg_step = g_tmpT.dot(x_t);
	Matrix dWg_step = g_tmpT.dot(h_prev);
	Matrix dbg_step = g_tmp.sum_cols();

	// output gate
	Matrix dq_step = dh_next * c_tan;
	tmp = (q_t * -1 + 1);
	Matrix dsigmoid_q = q_t * tmp;
	Matrix q_tmp = dq_step * dsigmoid_q;
	Matrix q_tmpT = q_tmp.transpose();
	Matrix dUq_step = q_tmpT.dot(x_t);
	Matrix dWq_step = q_tmpT.dot(h_prev);
	Matrix dbq_step = q_tmp.sum_cols();

	// input gate
	Matrix de_step = dc_next * e_t;
	tmp = (e_t * -1 + 1);
	Matrix dsigmoid_e = e_t * tmp;
	Matrix e_tmp = de_step * dsigmoid_e;
	Matrix e_tmpT = e_tmp.transpose();
	Matrix dUe_step = e_tmpT.dot(x_t);
	Matrix dWe_step = e_tmpT.dot(h_prev);
	Matrix dbe_step = e_tmp.sum_cols();

	// dh_prev
	res.dh_prev = (dh_next * c_tan * dsigmoid_q).dot(Wq);
	tmp = (dc_next * c_prev * dsigmoid_f).dot(Wf);
	res.dh_prev = res.dh_prev + tmp;
	tmp = (dc_next * g_t * dsigmoid_e).dot(We);
	res.dh_prev = res.dh_prev + tmp;
	tmp = (dc_next * e_t * dsigmoid_e).dot(Wg);
	res.dh_prev = res.dh_prev + tmp;

	res.dc_prev = f_t * dc_next;

	res.gradients = {
		{"We", dWe_step},
		{"Wf", dWf_step},
		{"Wg", dWg_step},
		{"Wq", dWq_step},
		{"Ue", dUe_step},
		{"Uf", dUf_step},
		{"Ug", dUg_step},
		{"Uq", dUq_step},
		{"be", dbe_step},
		{"bf", dbf_step},
		{"bg", dbg_step},
		{"bq", dbq_step},
	};

	return res;
}

LSTM_affine_fwd_res LSTM::affine_forward(vector<Matrix> h, Matrix U, Matrix b2) {
	// (np.matmul(h.reshape(N*T, Dh), U.T) + b2).reshape(N, T, V)
	Matrix UT = U.transpose();
	vector<Matrix> theta;
	for (Matrix& ht : h) {
		theta.push_back(ht.dot(UT));
	}

	return LSTM_affine_fwd_res{
		.theta = theta,
		.U = U,
		.b2 = b2,
		.h = h,
	};
}

LSTM_affine_bwd_res LSTM::affine_backward(vector<Matrix> dtheta, Matrix U_cached, Matrix b2_cached, vector<Matrix> h_cached) {
	return LSTM_affine_bwd_res{};
}

LSTM_fwd_res LSTM::forward(vector<Matrix> batch, Matrix h_init, Matrix c_init) {
	vector<Matrix> h(batch.size()); // vector of "h"s Matrixes from all time steps
	vector<LSTM_cell_fwd_res> progress;
	LSTM_cell_fwd_res init;
	init.c_next = c_init;
	init.h_next = h_init;
	progress.push_back(init);

	for (size_t t = 0; t < this->seq_len; t++) {
		Matrix x(batch.size(), this->vocab_size, 0);
		for (size_t n = 0; n < batch.size(); n++) {
			// set x[n,t] to row batch[n].row(t)
			x.row(t, batch[t].row(n));
		}


		LSTM_cell_fwd_res res = this->cell_forward(x, progress.back().h_next, progress.back().c_next);
		progress.push_back(res);
		h.push_back(res.h_next);
	}

	return LSTM_fwd_res{
		.h = h,
		.progress = vector<LSTM_cell_fwd_res>(progress.begin() + 1, progress.end()), // remove 0
	};
}

LSTM_bwd_res LSTM::backward(vector<Matrix> dh, vector<LSTM_cell_fwd_res> fwd_progress) {
	map<string,Matrix> gradients;

	return LSTM_bwd_res{
		.gradients_total = gradients,
	};
}

LSTM_training_res LSTM::train(vector<char> _X, size_t epochs, size_t batch_size, double lr) {
	int num_batches = _X.size() / this->seq_len;
	vector<char> X(_X.begin(), _X.begin() + num_batches * this->seq_len);
	vector<double> losses;



	Matrix U = Matrix::randn(this->vocab_size, this->n_h);
	Matrix b2 = Matrix(1, this->vocab_size, 0);


	for (size_t epoch = 0; epoch < epochs; epoch++) {
		cout << "Starting epoch no." << epoch << " of " << X.size() / this->seq_len
				 << " sequences" << endl;

		Matrix h_prev(batch_size, this->n_h, 0);
		Matrix c_prev(batch_size, this->n_h, 0);

		vector<Matrix> batch; // batch = N Matrixes of (T x V = vocab sizes) AKA 1 row = 1 one-hot encoding of a char
		for (size_t i = 0; i < batch_size; i++) {
			vector<double> _x, _y;
			for (size_t j = i; j < i + this->seq_len; j++) {
				auto encoding = this->one_hot_encode(this->char_to_idx[X[j]]);
				_x.insert(_x.end(), encoding.begin(), encoding.end());
			}
			Matrix x = Matrix(_x);
			x.reshape(this->seq_len, this->vocab_size);

			batch.push_back(x);
		}


		// forward pass
		LSTM_fwd_res fwd_pass = this->forward(batch, h_prev, c_prev);

		// affine layer
		this->affine_forward(fwd_pass.h, U, b2);


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
	// for (size_t i = 0; i < size; i++) {
	//     auto res = this->cell_forward(x, h, c);
	//     vector<double> probabilities = res.y_hat.ravel();
	//     h = res.h;
	//     c = res.c;
    //
	//     std::discrete_distribution<int> distribution(probabilities.begin(),
	//                                                                                              probabilities.end());
	//     const size_t idx = distribution(this->sample_random_generator);
    //
	//     sample += this->idx_to_char[idx];
	// }

	return sample;
}
