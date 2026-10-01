// CodinGame referee loop. Build a submittable single file with codingame/build_bundle.py.
#include <chrono>
#include <climits>
#include <iostream>
#include <string>

#include "board.h"
#include "search.h"

int main()
{
    int id; // id of your player.
    std::cin >> id; std::cin.ignore();
    int board_size;
    std::cin >> board_size; std::cin.ignore();

    Board* board = nullptr;
    // game loop
    while (1) {
        char grid[8][8];
        for (int i = 0; i < board_size; i++) {
            std::string line; // rows from top to bottom (viewer perspective).
            std::cin >> line; std::cin.ignore();
            for (int j = 0; j < 8; j++) {
                grid[i][j] = line[j];
            }
        }

        if (board == nullptr) {
            board = new Board(grid, id);
        }

        else {
            // in EXPERT mode the referee also sends the opponent's last move(s)
            std::string opponent_moves;
            std::cin >> opponent_moves;
            int tmp = opponent_moves[0] - 'a';
            if (tmp < 0 || tmp >= 8) {
                board = board->advance_move(-1, -1);
            }

            else {
                for (int i = 0; i < (int)opponent_moves.size(); i += 3) {
                    if (i > 0) {
                        board->advance_move(-1, -1);
                    }

                    int x = opponent_moves[i + 1] - '0' - 1;
                    int y = opponent_moves[i] - 'a';
                    board = board->advance_move(x, y);
                }
            }
        }

        int action_count; // number of legal actions for this turn.
        std::cin >> action_count; std::cin.ignore();
        for (int i = 0; i < action_count; i++) {
            std::string action; // the action
            std::cin >> action; std::cin.ignore();
        }

        auto [x, y] = get_best_move(board, 0.147, INT_MAX, true);
        board = board->advance_move(x, y);
        std::cout << "EXPERT " << (char)(y + 'a') << x + 1 << std::endl; // a-h1-8
    }
}
