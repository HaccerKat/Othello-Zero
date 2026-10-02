// WebAssembly entry point for the browser demo (web/). Built with Emscripten, see CMakeLists.txt.
#include <cstdio>
#include <cstring>
#include <string>

#include "board.h"
#include "search.h"

extern "C" {

// position: 64 board chars + side to move, as in engine/cli.cpp. max_depth <= 0 searches until
// time_ms runs out; otherwise the search stops after that depth (used by tests).
// Returns "<move> <eval>", e.g. "d3 0.1234", where eval is the searched value of the chosen move
// in [-1, 1] from black's perspective, or "pass 0".
const char* best_move(const char* position, int time_ms, int max_depth) {
    static std::string result;
    char grid[8][8];
    memcpy(grid, position, 64);
    Board board(grid, position[64] == '1');
    auto [x, y] = get_best_move(&board, time_ms / 1000.0, max_depth > 0 ? max_depth : INT_MAX);

    char buffer[32];
    if (x < 0) {
        snprintf(buffer, sizeof(buffer), "pass 0");
    }
    else {
        float eval = board.advance_move(x, y)->get_eval();
        snprintf(buffer, sizeof(buffer), "%c%d %.4f", 'a' + y, x + 1, eval);
    }
    result = buffer;
    return result.c_str();
}

}
