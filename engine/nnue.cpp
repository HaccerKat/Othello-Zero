#include "nnue.h"
#include <algorithm>
#include <cstring>
#include <math.h>

// BEGIN WEIGHTS (codingame/build_bundle.py replaces this block with an encoded string)
int16_t QUANT_WEIGHTS[WEIGHTS_SZ] = {
#include "weights/nnue_weights.inc"
};
int16_t QUANT_BIASES[BIASES_SZ] = {
#include "weights/nnue_biases.inc"
};
// END WEIGHTS

float nnue_evaluate(const char grid[8][8], bool player) {
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
