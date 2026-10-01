// Engine regression tests: move generation (perft), NNUE evaluation and search results.
//
//   engine_tests DATA_FILE
//
// DATA_FILE lists positions with the move the original CodinGame submission chose at depth 6
// and the evaluation from tools/nnue_reference.py.
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "board.h"
#include "nnue.h"
#include "search.h"

static int failures = 0;

static void check(bool ok, const std::string& message) {
    if (!ok) {
        failures++;
        std::cerr << "FAIL: " << message << "\n";
    }
}

static Board* make_board(const std::string& position) {
    char grid[8][8];
    memcpy(grid, position.data(), 64);
    return new Board(grid, position[64] == '1');
}

static std::string move_to_string(std::pair<int, int> move) {
    if (move.first < 0) return "pass";
    return std::string(1, 'a' + move.second) + std::to_string(move.first + 1);
}

// Leaf count at `depth` plies; a pass counts as a ply and finished games are leaves.
static long long perft(Board* board, int depth) {
    if (depth == 0 || board->find_if_game_ends()) return 1;
    long long leaves = 0;
    for (auto [child, move] : board->next_boards) {
        leaves += perft(child, depth - 1);
    }
    return leaves;
}

static void test_perft() {
    const std::string start = std::string(27, '.') + "10" + std::string(6, '.') + "01" + std::string(27, '.') + "0";
    const long long expected[] = {1, 4, 12, 56, 244, 1396, 8200, 55092};
    for (int depth = 1; depth <= 7; depth++) {
        Board* board = make_board(start);
        long long leaves = perft(board, depth);
        check(leaves == expected[depth], "perft(" + std::to_string(depth) + ") = " + std::to_string(leaves) +
                                         ", expected " + std::to_string(expected[depth]));
        delete board;
    }
}

static void test_positions(const char* path) {
    std::ifstream file(path);
    check(file.good(), std::string("cannot open ") + path);
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string position, expected_move;
        float expected_eval;
        fields >> position >> expected_move >> expected_eval;
        count++;

        char grid[8][8];
        memcpy(grid, position.data(), 64);
        float eval = nnue_evaluate(grid, position[64] == '1');
        check(std::fabs(eval - expected_eval) < 1e-5,
              position + ": eval " + std::to_string(eval) + ", expected " + std::to_string(expected_eval));

        Board* board = make_board(position);
        std::string move = move_to_string(get_best_move(board, 1e9, 6));
        check(move == expected_move, position + ": move " + move + ", expected " + expected_move);
        delete board;
    }
    check(count > 0, "no positions in data file");
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: engine_tests DATA_FILE\n";
        return 2;
    }

    test_perft();
    test_positions(argv[1]);
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all engine tests passed\n";
}
