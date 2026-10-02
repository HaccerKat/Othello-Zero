#pragma once
#include <chrono>
#include <climits>
#include <utility>

#include "board.h"

using Clock = std::chrono::steady_clock;

// Alpha-beta over the lazily built tree. Each node's eval is overwritten with its searched value,
// so children are ordered by the previous iteration's results.
void minimax(Board* position, int depth, float alpha, float beta, Clock::time_point start, double response_time);

// Iterative deepening from depth 1 until response_time (seconds) runs out. Returns (row, col),
// or (-1, -1) to pass. With max_depth set, stops after the depth max_depth iteration instead,
// which makes the result independent of machine speed.
std::pair<int, int> get_best_move(Board* position, double response_time, int max_depth = INT_MAX, bool verbose = false);
