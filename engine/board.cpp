#include "board.h"
#include "nnue.h"
#include <cstring>

Board::~Board() {
    for (auto [child, pair] : next_boards) {
        delete child;        // Recursively delete children
    }
}

// 0 -> black, 1 -> white
Board::Board(const char gr[8][8], bool p) {
    memcpy(grid, gr, sizeof(grid));
    player = p;
}

char Board::get_pos(int x, int y) const {
    return grid[x][y];
}

std::pair<int, int> Board::get_points() const {
    int black_points = 0;
    int white_points = 0;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] == '0') {
                black_points++;
            }
            if (grid[i][j] == '1') {
                white_points++;
            }
        }
    }

    return {black_points, white_points};
}

// 0 -> black wins, 1 -> draw, 2 -> white wins
int Board::get_winner_num() const {
    auto [black_points, white_points] = get_points();
    if (black_points == white_points) return 1;
    return black_points > white_points ? 0 : 2;
}

bool has_legal_move(const char grid[8][8], int who) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] != '.') {
                continue;
            }

            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }

                    // a legal move flanks a line of opponent discs with one of our own
                    int x = i + dx, y = j + dy, flip_number = 0;
                    while (x >= 0 && x < 8 && y >= 0 && y < 8 && grid[x][y] != '.' && grid[x][y] - '0' != who) {
                        x += dx, y += dy, flip_number++;
                    }

                    if (flip_number > 0 && x >= 0 && x < 8 && y >= 0 && y < 8 && grid[x][y] - '0' == who) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

void Board::get_static_eval() {
    // Most evaluated positions are leaves that are never searched, so avoid generating
    // their children just to find out whether the game is over.
    bool ends = found_next_moves ? find_if_game_ends()
                                 : !has_legal_move(grid, player) && !has_legal_move(grid, player ^ 1);
    if (ends) {
        auto [black_points, white_points] = get_points();
        if (black_points == white_points) eval = DRAW;
        else eval = black_points > white_points ? BLACK_WINS : WHITE_WINS;
        return;
    }

    eval = nnue_evaluate(grid, player);
}

float Board::get_eval() {
    // static eval has not been calculated yet
    if (eval == -2.0) {
        get_static_eval();
    }

    return eval;
}

void Board::change_eval(float new_eval) {
    eval = new_eval;
}

void Board::find_next_boards() {
    if (found_next_moves) {
        return;
    }

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (grid[i][j] != '.') {
                continue;
            }

            char next_grid[8][8];
            memcpy(next_grid, grid, sizeof(next_grid));
            next_grid[i][j] = player + '0';
            bool can_flip = 0;
            for (int dx = -1; dx <= 1; dx++) {
                for (int dy = -1; dy <= 1; dy++) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }

                    int x = i + dx, y = j + dy;
                    int flip_number = 0;
                    while (x >= 0 && x < 8 && y >= 0 && y < 8) {
                        if (grid[x][y] == '.') {
                            break;
                        }

                        int cell = grid[x][y] - '0';
                        if (cell == player) {
                            if (flip_number > 0) {
                                can_flip = 1;
                                x = i + dx, y = j + dy;
                                while (true) {
                                    cell = grid[x][y] - '0';
                                    if (cell == player) {
                                        break;
                                    }

                                    else {
                                        next_grid[x][y] = player + '0';
                                    }

                                    x += dx, y += dy;
                                }
                            }

                            break;
                        }

                        flip_number++, x += dx, y += dy;
                    }
                }
            }

            if (can_flip) {
                Board* next_board = new Board(next_grid, player ^ 1);
                next_boards.push_back({next_board, {i, j}});
            }
        }
    }

    found_next_moves = 1;
    if (next_boards.empty()) {
        skip_turn = 1;
        Board* next_board = new Board(grid, player ^ 1);
        next_boards.push_back({next_board, {-1, -1}});
    }
}

bool Board::find_if_game_ends() {
    find_next_boards();
    if (game_ends != -1) {
        return game_ends;
    }

    if (skip_turn) {
        Board* other_player_board = next_boards[0].first;
        other_player_board->find_next_boards();
        game_ends = other_player_board->has_no_move();
        return game_ends;
    }

    else {
        game_ends = 0;
        return game_ends;
    }
}

// Returns the child reached by the move, or nullptr if the move is illegal. Use (-1, -1) to pass.
Board* Board::advance_move(int input_x, int input_y) {
    find_next_boards();
    for (auto game_state : next_boards) {
        auto [x, y] = game_state.second;
        if (input_x == x && input_y == y) {
            return game_state.first;
        }
    }

    return nullptr;
}
