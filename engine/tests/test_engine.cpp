// Engine regression tests: move generation (perft), NNUE evaluation and search results.
//
//   engine_tests DATA_FILE
//
// DATA_FILE lists positions with the move the original CodinGame submission chose at depth 6
// and the evaluation from tools/nnue_reference.py.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>

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

// The original dense forward pass, kept verbatim as a reference: optimizations to nnue_evaluate
// must give bit-identical results.
static float reference_evaluate(const char grid[8][8], bool player) {
    int nnue_layer[128];
    memset(nnue_layer, 0, sizeof(nnue_layer));
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int pos = i * 8 + j;
            if (grid[i][j] != '.') {
                if (grid[i][j] - '0' == player) nnue_layer[pos] = 1;
                else nnue_layer[pos + 64] = 1;
            }
        }
    }

    int16_t res_int[RES_SZ];
    for (int i = 0; i < LAYERS[0]; i++) {
        res_int[i] = round(nnue_layer[i] * QUANT_MULT);
    }

    float eval = 0;
    int idx_weights = 0, idx_biases = 0, idx_res = LAYERS[0];
    for (int i = 1; i < CNT_LAYERS; i++) {
        int start_idx = idx_res - LAYERS[i - 1];
        for (int j = 0; j < LAYERS[i]; j++) {
            int acc = 0;
            for (int k = 0; k < LAYERS[i - 1]; k++) {
                acc += QUANT_WEIGHTS[idx_weights + k] * res_int[start_idx + k];
            }

            res_int[idx_res] = round(acc / QUANT_MULT) + QUANT_BIASES[idx_biases];
            if (i + 1 == CNT_LAYERS) {
                float val = res_int[idx_res] / QUANT_MULT;
                eval = tanh(val) * (player == 1 ? -1 : 1);
            }
            else {
                res_int[idx_res] = std::max((int16_t)0, res_int[idx_res]);
            }

            idx_weights += LAYERS[i - 1], idx_biases++, idx_res++;
        }
    }

    return eval;
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

// With 7 empty squares, black's only non-losing move is h4, which draws.
static void test_endgame_draw() {
    Board* board = make_board("1.1.001.11.10000111010011111000.11111111110000011111100110.000000");
    std::string move = move_to_string(get_best_move(board, 1e9, 12));
    check(move == "h4", "endgame draw: move " + move + ", expected h4");
    delete board;
}

// Compares nnue_evaluate with the reference on each data file position and every position up to
// two plies after it (passes included).
static void test_eval_exact(const char* path) {
    std::ifstream file(path);
    std::vector<Board*> roots, boards;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        Board* root = make_board(line.substr(0, 65));
        roots.push_back(root);
        boards.push_back(root);
        root->find_next_boards();
        for (auto [child, move] : root->next_boards) {
            boards.push_back(child);
            child->find_next_boards();
            for (auto [grandchild, child_move] : child->next_boards) boards.push_back(grandchild);
        }
    }

    int mismatches = 0;
    for (Board* board : boards) {
        char grid[8][8];
        for (int i = 0; i < 64; i++) grid[i / 8][i % 8] = board->get_pos(i / 8, i % 8);
        if (nnue_evaluate(grid, board->get_player()) != reference_evaluate(grid, board->get_player())) mismatches++;
    }
    check(boards.size() > 1000, "too few positions for the exact eval test: " + std::to_string(boards.size()));
    check(mismatches == 0, "nnue_evaluate differs from the reference on " + std::to_string(mismatches) + " of " +
                           std::to_string(boards.size()) + " positions");

    for (Board* root : roots) delete root;  // children are owned by their roots
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
    test_endgame_draw();
    test_eval_exact(argv[1]);
    test_positions(argv[1]);
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "all engine tests passed\n";
}
