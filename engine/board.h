#pragma once
#include <utility>
#include <vector>

// A node of the game tree. Children are generated lazily and owned by their parent.
// Grid cells: '.' empty, '0' black, '1' white. player: 0 = black to move, 1 = white.
// A pass is represented as a child with move (-1, -1).
class Board {
    char grid[8][8];
    bool player, found_next_moves = 0, skip_turn = 0;
    int game_ends = -1;
    float eval = -2.0;
public:
    const float BLACK_WINS = 1, WHITE_WINS = -1, DRAW = 0.0;
    std::vector<std::pair<Board*, std::pair<int, int>>> next_boards;
    ~Board();
    Board(const char gr[8][8], bool p);
    char get_pos(int x, int y) const;
    bool has_no_move() const {return skip_turn;}
    bool get_player() const {return player;}
    std::pair<int, int> get_points() const;
    int get_winner_num() const;
    void get_static_eval();
    float get_eval();
    void change_eval(float new_eval);
    void find_next_boards();
    bool find_if_game_ends();
    Board* advance_move(int input_x, int input_y);
};
