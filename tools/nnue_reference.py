"""NumPy reference for the engine's quantized NNUE forward pass (engine/nnue.cpp).

It reproduces the C++ integer and float32 arithmetic step by step, so evaluations should
agree to within float rounding of the final tanh.

Usage: python tools/nnue_reference.py < positions.txt
    prints one evaluation per position (positive = good for black)
"""

import sys
from pathlib import Path

import numpy as np

from weights_codec import read_inc

LAYERS = [128, 256, 32, 32, 1]
QUANT_MULT = np.float32(46.5 / 0.05)
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "engine" / "weights"


def load_layers(weights_dir=WEIGHTS_DIR):
    """Split the flat weight/bias arrays into per-layer (out, in) matrices and bias vectors."""
    weights = np.array(read_inc(weights_dir / "nnue_weights.inc"), dtype=np.int64)
    biases = np.array(read_inc(weights_dir / "nnue_biases.inc"), dtype=np.int64)
    layers, w, b = [], 0, 0
    for n_in, n_out in zip(LAYERS, LAYERS[1:]):
        layers.append((weights[w:w + n_in * n_out].reshape(n_out, n_in), biases[b:b + n_out]))
        w, b = w + n_in * n_out, b + n_out
    return layers


def c_round(x):
    """C's round(): half away from zero."""
    return np.sign(x) * np.floor(np.abs(x) + np.float32(0.5))


def evaluate(position, layers):
    board, player = position[:64], int(position[64])
    inputs = np.zeros(128, dtype=np.int64)
    for i, cell in enumerate(board):
        if cell != ".":
            inputs[i if int(cell) == player else i + 64] = 1

    x = c_round(inputs.astype(np.float32) * QUANT_MULT).astype(np.int64)
    for index, (weight, bias) in enumerate(layers):
        acc = weight @ x
        y = c_round(acc.astype(np.float32) / QUANT_MULT) + bias.astype(np.float32)
        if np.any(np.abs(y) > 32767):
            raise OverflowError("activation does not fit in int16")
        x = y.astype(np.int64)
        if index + 1 < len(layers):
            x = np.maximum(x, 0)

    value = np.tanh(np.float32(x[0]) / QUANT_MULT)
    return float(value if player == 0 else -value)


if __name__ == "__main__":
    layers = load_layers()
    for line in sys.stdin:
        line = line.strip()
        if line:
            print(f"{evaluate(line[:65], layers):.7f}")
