#include "log.h"
#include "lstm.h"
#include "matrix.h"
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
#include "stacktrace.h"

void print_help(string);
vector<char> read_dataset(string &);

using namespace std;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  string dataset_filepath;
  string losses_output_file;

  int c;
  while (1) {
    static struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                           {"filepath", no_argument, 0, 'f'},
                                           {"losses-out", no_argument, 0, 'l'},
                                           {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "hf:l:", long_options, &option_index);

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
      losses_output_file = string(optarg);
      break;

    case 'h':
      print_help(argv[0]);
      return 0;
      break;
    }
  }

  // try {
  //   vector<double> v(12);
  //   int n=0;
  //   std::generate(v.begin(), v.end(), [&]{ return ++n; });
  //   Matrix a(v);
  //   a.print();
  //   a.reshape(9, 1);
  //   a.print();
  // } catch (char const *e) {
  //   cerr << e << endl;
  // }
  // return 0;

  if (dataset_filepath == "") {
	dataset_filepath = "./data/HP1-short.txt";
    // cerr << "Invalid dataset filepath";
    // return 1;
  }

  vector<char> data = read_dataset(dataset_filepath);
  for (char &c : data) {
    c = tolower(c);
  }

  set<char> chars(data.begin(), data.end());
  size_t vocab_size = chars.size();

  cout << "data has " << data.size() << " characters, " << vocab_size
       << " unique" << endl;

  map<char, size_t> char_to_idx;
  map<size_t, char> idx_to_char;
  int chars_i = 0;
  for (char c : chars) {
    char_to_idx[c] = chars_i;
    idx_to_char[chars_i] = c;

	// cout << c << ":" << chars_i << endl;

	chars_i++;
}

  try {
    LSTM nn = LSTM(char_to_idx, idx_to_char, vocab_size, 25, 200);
	nn.train(data, 10, 0.1);

    // Matrix h_prev(25, 1, 0);
    // Matrix c_prev(25, 1, 0);
	// cout << nn.sample(h_prev, c_prev, 100) << endl;
  } catch (char const *e) {
    cerr << e << endl;
  }

  return 0;
}

void print_help(string binary_path) {
  cout << binary_path << " [-h]" << endl;
  cout << "  -h --help\t\tShow this help" << endl;
  cout << "  -f --filepath\t\tPath to dataset file for training" << endl;
}

vector<char> read_dataset(string &filename) {
  ifstream ifs(filename.c_str(), ios::in | ios::binary | ios::ate);

  ifstream::pos_type fileSize = ifs.tellg();
  ifs.seekg(0, ios::beg);

  vector<char> bytes(fileSize);
  ifs.read(bytes.data(), fileSize);

  return bytes;
}
