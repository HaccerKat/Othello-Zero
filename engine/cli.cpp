// Native command-line engine.
//
//   othello-engine [--time SECONDS] [--depth N] [--verbose]
//
// Reads one position per line from stdin: 64 board chars (row-major from the top-left,
// '.' empty, '0' black, '1' white) followed by the side to move ('0' or '1'), optionally
// separated by whitespace. Prints the chosen move per line, e.g. "d3", or "pass".
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "board.h"
#include "search.h"

static void usage() {
    std::cerr << "usage: othello-engine [--time SECONDS] [--depth N] [--verbose]\n";
    std::exit(2);
}

int main(int argc, char** argv) {
    double response_time = 0.15;
    int max_depth = INT_MAX;
    bool verbose = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--time" && i + 1 < argc) {
            response_time = std::atof(argv[++i]);
        }
        else if (arg == "--depth" && i + 1 < argc) {
            max_depth = std::atoi(argv[++i]);
            response_time = 1e9;
        }
        else if (arg == "--verbose") {
            verbose = true;
        }
        else {
            usage();
        }
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        std::string data;
        for (char c : line) {
            if (c == '.' || c == '0' || c == '1') data.push_back(c);
        }
        if (data.empty()) continue;
        if (data.size() != 65) {
            std::cerr << "expected 64 board chars and a side to move, got: " << line << "\n";
            return 1;
        }

        char grid[8][8];
        memcpy(grid, data.data(), 64);
        Board board(grid, data[64] == '1');
        auto [x, y] = get_best_move(&board, response_time, max_depth, verbose);
        if (x < 0) std::cout << "pass" << std::endl;
        else std::cout << (char)('a' + y) << x + 1 << std::endl;
    }
}
