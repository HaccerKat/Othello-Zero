#pragma once
#include <cstdint>

// Distilled value network: 128 -> 256 -> 32 -> 32 -> 1, ReLU between layers, tanh output.
// Inputs are one-hot squares from the side to move's perspective: [0, 64) own discs, [64, 128) opponent discs.
// Weights are int16, quantized with scale QUANT_MULT; every layer runs in integer arithmetic.
constexpr int CNT_LAYERS = 5;
constexpr int LAYERS[CNT_LAYERS] = {128, 256, 32, 32, 1};
constexpr int WEIGHTS_SZ = 42016, BIASES_SZ = 321, RES_SZ = 449;
constexpr float BOUND = 0.05, QUANT_MULT = 46.5 / BOUND;

extern int16_t QUANT_WEIGHTS[WEIGHTS_SZ], QUANT_BIASES[BIASES_SZ];

// Evaluation in [-1, 1] from black's perspective (positive = good for black).
float nnue_evaluate(const char grid[8][8], bool player);
