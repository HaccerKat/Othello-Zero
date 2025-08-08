// For simulating lots of games at a time

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>
#include <tuple>
#include <cmath>
#include <chrono>

#include "board.h"
#include "experimental_board.h"
#include "minimax.h"
#include "global_funcs.h"

#ifdef DEBUG
#include "algo/debug.hpp"
#else
#define dbg(...)
#define dbgm(...)
#define ulim_stack()
#endif

int simulate_game() {
    char grid[8][8];
    memset(grid, '.', sizeof(grid));
    grid[3][3] = grid[4][4] = '1';
    grid[3][4] = grid[4][3] = '0';
    Board* board = new Board(grid, 0);
    int init_nnue_layer[LAYERS[0]];
    // int16_t init_layer_2[LAYERS[1]];
    memset(init_nnue_layer, 0, sizeof(init_nnue_layer));
    // for (int i = 0; i < LAYERS[1]; i++) {
    //     init_layer_2[i] = QUANT_BIASES[i];
    // }

    Experimental_Board* experimental_board = new Experimental_Board(grid, 0);

    // Board* orig_board = board;
    // Experimental_Board* orig_experimental_board = experimental_board;
    // 0 - board first (black), 1 - experimental first
    int starting_player = rnd(0, 1), current_player = starting_player;
    while (!board->find_if_game_ends()) {
        int x, y;
        if (current_player == 0) {
            board->find_next_boards();
            std::tie(x, y) = get_best_move<Board*, int>(board, 0.03, 0);
        }

        else {
            experimental_board->find_next_boards();
            std::tie(x, y) = get_best_move<Experimental_Board*, float>(experimental_board, 0.03, 1);
        }

        // std::vector<int> prev_nnue_layer_vec = experimental_board->get_nnue_layer();
        Board* new_board = board->advance_move(x, y);
        // std::vector<int16_t> layer_2_vec = experimental_board->get_layer_2();
        // int prev_nnue_layer[LAYERS[0]];
        // for (int i = 0; i < LAYERS[0]; i++) {
        //     prev_nnue_layer[i] = prev_nnue_layer_vec[i];
        // }
        //
        // int16_t layer_2[LAYERS[1]];
        // for (int i = 0; i < LAYERS[1]; i++) {
        //     layer_2[i] = layer_2_vec[i];
        // }

        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 8; j++) {
                grid[i][j] = new_board->get_pos(i, j);
            }
        }

        bool player = new_board->get_player();
        delete board;
        delete experimental_board;
        board = new Board(grid, player);
        experimental_board = new Experimental_Board(grid, player);
        current_player ^= 1;
    }

    int winner = board->get_winner_num();
    // delete orig_board;
    // delete orig_experimental_board;
    if (winner == 1) return 2;
    return ((winner / 2) ^ starting_player);
}

int32_t main() {
    // . - empty square
    // * - valid move square
    // 0 - black piece
    // 1 - white piece
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Enter NNUE Name: ";
    std::string nnue_name;
    std::cin >> nnue_name;
    std::ifstream weights_file("./models/weights_" + nnue_name + ".txt");
    std::ifstream biases_file("./models/biases_" + nnue_name + ".txt");

    double temp;
    int temp_idx = 0;
    while (weights_file >> temp) {
        WEIGHTS[temp_idx] = temp;
        QUANT_WEIGHTS[temp_idx++] = round(temp * QUANT_MULT);
    }

    temp_idx = 0;
    while (biases_file >> temp) {
        BIASES[temp_idx] = temp;
        QUANT_BIASES[temp_idx++] = round(temp * QUANT_MULT);
    }

    weights_file.close();
    biases_file.close();

    for (int i = 0; i < LAYERS[0] * LAYERS[1]; i++) {
        int x = i % LAYERS[0], y = i / LAYERS[0];
        QUANT_WEIGHTS_LAYER_0[LAYERS[1] * x + y] = QUANT_WEIGHTS[i];
    }

    // std::ifstream weights_file_2("./models/weights_" + nnue_name + "_quantized.txt");
    // std::ifstream biases_file_2("./models/biases_" + nnue_name + "_quantized.txt");
    // for (int i = 0; i < WEIGHTS_SZ; i++) {
    //     char c;
    //     weights_file_2 >> c;
    //     int x = c - '!' - 46;
    //     if (c == '~') {
    //         weights_file_2 >> c;
    //         x = (c - '!' - 46) * 93;
    //         weights_file_2 >> c;
    //         x += (c - '!');
    //     }
    //
    //     QUANT_WEIGHTS[i] = x;
    // }
    //
    // for (int i = 0; i < BIASES_SZ; i++) {
    //     char c;
    //     int x = c - '!' - 46;
    //     biases_file_2 >> c;
    //     if (c == '~') {
    //         biases_file_2 >> c;
    //         x = (c - '!' - 46) * 93;
    //         biases_file_2 >> c;
    //         x += (c - '!');
    //     }
    //
    //     QUANT_BIASES[i] = x;
    // }
    //
    // weights_file_2.close();
    // biases_file_2.close();

    // with multithreading
    constexpr int num_threads = 16;
    const int num_games = 100;
    int positions_generated = 0;
    // auto start = std::chrono::steady_clock::now();
    int board_wins = 0, experimental_wins = 0;
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_games; i++) {
        int winner = simulate_game();
        #pragma omp critical
        {
            if (winner == 1) experimental_wins++;
            if (winner == 0) board_wins++;
        }
    }

    int num_non_draws = board_wins + experimental_wins;
    double probability = 1.0 * experimental_wins / num_non_draws;
    double error = 1.96 * sqrt(probability * (1 - probability) / num_non_draws);
    double lower_bound = 100.0 * (probability - error), upper_bound = 100.0 * (probability + error);
    std::cout << "Board Wins: " << board_wins << "\n";
    std::cout << "Experimental Wins: " << experimental_wins << "\n";
    std::cout << "Experimental WR: " << 100.0 * probability << "%\n";
    std::cout << "95% Confidence Interval: [" << lower_bound << "%, " << upper_bound << "%]\n";
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Runtime: " << elapsed.count() << "s\n";
}

/*
Equal Footing
Board Wins: 272
Experimental Wins: 223
Experimental WR: 45.0505%
95% Confidence Interval: [40.6674%, 49.4336%]
Runtime: 177.267s

Board gets 3/10 the time
Board Wins: 177
Experimental Wins: 320
Experimental WR: 64.3863%
95% Confidence Interval: [60.1763%, 68.5963%]
Runtime: 116.447s

Experimental gets 3/10 the time
Board Wins: 311
Experimental Wins: 188
Experimental WR: 37.6754%
95% Confidence Interval: [33.4236%, 41.9271%]
Runtime: 117.5s
*/