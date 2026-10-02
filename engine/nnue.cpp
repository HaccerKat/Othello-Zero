#include "nnue.h"
#include <algorithm>
#include <math.h>

// BEGIN WEIGHTS (codingame/build_bundle.py replaces this block with an encoded string)
int16_t QUANT_WEIGHTS[WEIGHTS_SZ] = {
#include "weights/nnue_weights.inc"
};
int16_t QUANT_BIASES[BIASES_SZ] = {
#include "weights/nnue_biases.inc"
};
// END WEIGHTS

// An occupied square's input quantizes to round(1 * QUANT_MULT), an empty one's to 0.
constexpr int INPUT_SCALE = 930;
static_assert(QUANT_MULT == INPUT_SCALE, "update INPUT_SCALE to round(QUANT_MULT)");

// First-layer weights transposed so each input's 256 weights are contiguous.
struct ColumnTable {
    int16_t columns[LAYERS[0]][LAYERS[1]];
};

static ColumnTable build_columns() {
    ColumnTable table;
    for (int j = 0; j < LAYERS[1]; j++) {
        for (int i = 0; i < LAYERS[0]; i++) {
            table.columns[i][j] = QUANT_WEIGHTS[j * LAYERS[0] + i];
        }
    }

    return table;
}

float nnue_evaluate(const char grid[8][8], bool player) {
    // built on first use, after the CodinGame bundle has decoded the weights
    static const ColumnTable table = build_columns();

    // First layer: the inputs are 0 or INPUT_SCALE, so the dot product is exactly
    // INPUT_SCALE * (sum of the weights of occupied squares). Summing only those columns
    // gives the same integer as the dense product, so the result is bit-identical.
    int32_t sum[LAYERS[1]] = {};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int pos = i * 8 + j;
            if (grid[i][j] != '.') {
                const int16_t* column = table.columns[grid[i][j] - '0' == player ? pos : pos + 64];
                for (int k = 0; k < LAYERS[1]; k++) {
                    sum[k] += column[k];
                }
            }
        }
    }

    int16_t res_int[RES_SZ];
    int idx_res = LAYERS[0];
    for (int j = 0; j < LAYERS[1]; j++) {
        int acc = sum[j] * INPUT_SCALE;
        res_int[idx_res] = round(acc / QUANT_MULT) + QUANT_BIASES[j];
        // ReLU
        res_int[idx_res] = std::max((int16_t)0, res_int[idx_res]);
        idx_res++;
    }

    float eval = 0;
    int idx_weights = LAYERS[0] * LAYERS[1], idx_biases = LAYERS[1];
    for (int i = 2; i < CNT_LAYERS; i++) {
        // index of the start of the previous layer
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

            // ReLU
            else {
                res_int[idx_res] = std::max((int16_t)0, res_int[idx_res]);
            }

            idx_weights += LAYERS[i - 1], idx_biases++, idx_res++;
        }
    }

    return eval;
}
