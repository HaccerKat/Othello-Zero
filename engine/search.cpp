#include "search.h"
#include <algorithm>
#include <iostream>

void minimax(Board* position, int depth, float alpha, float beta, Clock::time_point start, double response_time) {
    std::chrono::duration<double> elapsed = Clock::now() - start;
    if (depth == 0 || position->find_if_game_ends() || elapsed.count() > response_time) {
        return;
    }

    position->find_next_boards();
    std::sort(position->next_boards.begin(), position->next_boards.end(), [&](std::pair<Board*, std::pair<int, int>> a, std::pair<Board*, std::pair<int, int>> b) {
        if (!position->get_player()) {
            return a.first->get_eval() > b.first->get_eval();
        }

        else {
            return a.first->get_eval() < b.first->get_eval();
        }
    });

    // black to move
    if (!position->get_player()) {
        position->change_eval(position->WHITE_WINS);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(std::max(position->get_eval(), child->get_eval()));
            alpha = std::max(alpha, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }

    // white to move
    else {
        position->change_eval(position->BLACK_WINS);
        for (auto [child, pair] : position->next_boards) {
            minimax(child, depth - 1, alpha, beta, start, response_time);
            position->change_eval(std::min(position->get_eval(), child->get_eval()));
            beta = std::min(beta, child->get_eval());
            if (beta <= alpha) {
                break;
            }
        }
    }
}

std::pair<int, int> get_best_move(Board* position, double response_time, int max_depth, bool verbose) {
    auto start = Clock::now();
    position->find_next_boards();
    int depth = 6;
    float eval = position->DRAW;
    // a legal move (or the pass) to fall back on if there is no time for a single iteration
    std::pair<int, int> move = position->next_boards[0].second;
    bool searched = false;
    while (std::chrono::duration<double>(Clock::now() - start).count() < response_time && depth < 64) {
        // pick the best move found by the last completed iteration
        searched = true;
        float best_eval = position->get_player() ? position->BLACK_WINS : position->WHITE_WINS;
        for (auto [child, pair] : position->next_boards) {
            // black to move
            if (!position->get_player()) {
                if (child->get_eval() >= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            }

            // white to move
            else {
                if (child->get_eval() <= best_eval) {
                    best_eval = child->get_eval(), move = pair;
                }
            }
        }

        eval = best_eval;
        if (depth > max_depth) {
            break;
        }

        minimax(position, depth, position->WHITE_WINS, position->BLACK_WINS, start, response_time);
        depth++;
    }

    if (!searched) {
        std::cerr << "Warning: out of time before searching, playing the first legal move" << std::endl;
    }

    if (verbose) {
        std::chrono::duration<double> elapsed = Clock::now() - start;
        std::cerr << "Depth: " << depth << std::endl;
        std::cerr << "Eval: " << eval << std::endl;
        std::cerr << "Elapsed Time: " << elapsed.count() << std::endl;
    }

    return move;
}
