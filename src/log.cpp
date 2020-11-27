#include <iostream>
#include "log.h"

using namespace std;

void Log::debug(string msg, string end) {
    cerr << "\033[1;34m[DBUG] " << msg << "\033[0m" << end;
}

void Log::info(string msg, string end) {
    cerr << "\033[1;36;1m[INFO] " << msg << "\033[0m" << end;
}

void Log::warn(string msg, string end) {
    cerr << "\033[1;33m[WARN] " << msg << "\033[0m" << end;
}

void Log::error(string msg, string end) {
    cerr << "\033[1;31m[EROR] " << msg << "\033[0m" << end;
}
