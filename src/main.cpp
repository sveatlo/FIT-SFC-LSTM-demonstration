#include "log.h"
#include "lstm.h"
#include "matrix.h"
#include "stacktrace.h"
#include <algorithm>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <map>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

void print_help(string);
vector<char> read_dataset(string &);

using namespace std;

int main(int argc, char **argv) {
	(void)argc;
	(void)argv;

	string dataset_filepath = "data/dinos.txt";
	double learning_rate = 0.01;
	int epochs = 10;
	int hidden_layers = 100;
	int sequence_len = 25;

	int c;
	while (1) {
		static struct option long_options[] = {
				{"help", no_argument, 0, 'h'},
				{"filepath", no_argument, 0, 'f'},
				{"learning-rate", no_argument, 0, 'l'},
				{"epochs", no_argument, 0, 'e'},
				{"hidden-layers", no_argument, 0, 'i'},
				{"sequence-length", no_argument, 0, 's'},
				{0, 0, 0, 0}};
		/* getopt_long stores the option index here. */
		int option_index = 0;

		c = getopt_long(argc, argv, "hf:l:e:i:s:", long_options, &option_index);

		/* Detect the end of the options. */
		if (c == -1)
			break;

		switch (c) {
		case 0:
			/* If this option set a flag, do nothing else now. */
			if (long_options[option_index].flag != 0)
				break;
			printf("option %s", long_options[option_index].name);
			if (optarg)
				printf(" with arg %s", optarg);
			printf("\n");
			break;

		case 'f':
			dataset_filepath = string(optarg);
			break;

		case 'l':
			learning_rate = stod(string(optarg));
			break;

		case 'e':
			try {
				epochs = stoi(optarg);
			} catch (...) {
				cout << "epochs must be integer";
				exit(1);
			}
			break;

		case 'i':
			try {
				hidden_layers = stoi(optarg);
			} catch (...) {
				cout << "number of hidden layers must be integer";
				exit(1);
			}
			break;

		case 's':
			try {
				sequence_len = stoi(optarg);
			} catch (...) {
				cout << "sequence length must be integer";
				exit(1);
			}
			break;

		case 'h':
			print_help(argv[0]);
			return 0;
			break;
		}
	}

	// Matrix<double> a = Matrix<double>::randn(2, 3);
	// Matrix<double> b(3,3,1);
	// a.vstack(b).print();
	// return 0;

	// Matrix<double> a(3,3,1);
	// Matrix<double> b(3,3,1);
	// (a+b).print();
	// return 0;

	// vector<double> ad = {3. , -1. ,  0.3,  0. ,  2. ,  1.};
	// vector<double> bd = { 1,  4,  5,  6,  8,  2, -3,  3,  9,  7,  1,  0 };
	// Matrix<double> a(ad);
	// a.reshape(2, 3);
	// Matrix<double> b(bd);
	// b.reshape(3, 4);
	// a.print();
	// b.print();
	// a.dot(b).print();
	// return 0;

	// vector<double> d = {0.003,-3,0,-0.000001,2,1};
	// Matrix<double> a;
	// Matrix<double> x = Matrix<double>(d);
	// x.reshape(2,3);
    //
	// cout << "\nsigmoid: " << endl;
	// a = ((x * -1).exp() + 1).divides(1);
	// a.print();
    //
	// cout << "\ndsigmoid: " << endl;
	// auto tmp = ((x * -1) + 1);
	// a = x * tmp;
	// a.print();
    //
	// cout << "\ntanh: " << endl;
	// x.tanh().print();
    //
	// cout << "\ndtanh: " << endl;
	// a = (x.pow(2) * -1) + 1;
	// a.print();
    //
	// cout << "\nsoftmax" << endl;
	// Matrix<double> e_x = x.exp();
	// a = e_x / (e_x.sum() + 1e-8);
	// a.print();
    //
	// cout << "\nexp: " << endl;
	// x.exp().print();
	// return 0;



	// START

	vector<char> data = read_dataset(dataset_filepath);
	for (char &c : data) {
		c = tolower(c);
	}

	set<char> chars(data.begin(), data.end());
	unsigned vocab_size = chars.size();

	cout << "data has " << data.size() << " characters, " << vocab_size
			 << " unique" << endl;

	map<char, unsigned> char_to_idx;
	map<unsigned, char> idx_to_char;
	int chars_i = 0;
	for (char c : chars) {
		char_to_idx[c] = chars_i;
		idx_to_char[chars_i] = c;

		// cout << c << ":" << chars_i << endl;

		chars_i++;
	}

	cout << "======================================================" << endl;
	cout << "Training with following parameters:" << endl;
	cout << "\tlearning rate: " << learning_rate << endl;
	cout << "\tno. of epochs: " << epochs << endl;
	cout << "\tnumber of hidden layers: " << hidden_layers << endl;
	cout << "\tsequence length: " << sequence_len << endl;
	cout << "\tvocabulary size: " << vocab_size << endl;
	cout << "\t  vocabulary: ";
	for (unsigned i = 0; i < chars.size(); i++) {
		cout << i << ":";
		switch (idx_to_char[i]) {
			case '\n':
				cout << "\\n";
				break;
			default:
				cout << idx_to_char[i];
		}
		cout << " ";
	}
	cout << endl;

	// try {
		LSTM nn =
				LSTM(char_to_idx, idx_to_char, vocab_size, hidden_layers, sequence_len);
		LSTM_training_res res = nn.train(data, epochs, learning_rate);
		cout << "================== Training finished =================" << endl;

		cout << "Losses progress: " << endl;
		for (auto &x : res.lossses) {
			cout << x << " ";
		}
		cout << endl << endl;

		Matrix<double> h_prev(hidden_layers, 1, 0);
		Matrix<double> c_prev(hidden_layers, 1, 0);
		cout << nn.sample(100, 'a') << endl;
	// } catch (char const *e) {
	//     cerr << e << endl;
	// }

	return 0;
}

void print_help(string binary_path) {
	cout << binary_path << " [-h]" << endl;
	cout << "	-h --help\t\tShow this help" << endl;
	cout << "	-f --filepath\t\tPath to dataset file for training" << endl;
	cout << "	-l --learning-rate\t\tLearning rate for training" << endl;
	cout << "	-e --epochs\t\tNumber of epochs for training" << endl;
	cout << "	-i --hidden-layers\t\tNumber of hidden layers" << endl;
	cout << "	-s --sequence-length\t\tSequence length" << endl;
}

vector<char> read_dataset(string &filename) {
	ifstream ifs(filename.c_str(), ios::in | ios::binary | ios::ate);

	ifstream::pos_type fileSize = ifs.tellg();
	ifs.seekg(0, ios::beg);

	vector<char> bytes(fileSize);
	ifs.read(bytes.data(), fileSize);

	return bytes;
}
