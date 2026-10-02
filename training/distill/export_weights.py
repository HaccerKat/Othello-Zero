"""Quantize a trained NNUE checkpoint and export it for the engine.

Usage: python export_weights.py CHECKPOINT [--out DIR]

Weights and biases are scaled by QUANT_MULT and rounded to int16, then written as
nnue_weights.inc / nnue_biases.inc in DIR (default: engine/weights), which the engine
compiles in and codingame/build_bundle.py encodes for CodinGame.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from model import NeuralNetworkNNUE, load_model

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from weights_codec import write_inc  # noqa: E402

# Must match engine/nnue.h
BOUND = 0.05
QUANT_MULT = 46.5 / BOUND
LAYERS = ["layer1", "layer2", "layer3", "value"]


def quantize(values):
    quantized = [round(x * QUANT_MULT) for x in values]
    if max(abs(q) for q in quantized) > 32767:
        raise OverflowError("a weight does not fit in int16 at this BOUND")
    return quantized


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", default=str(ROOT / "engine" / "weights"))
    args = parser.parse_args()

    model = load_model(NeuralNetworkNNUE, args.checkpoint, device="cpu")
    params = {k: v.numpy() for k, v in model.state_dict().items()}
    # Linear weights are (out, in), flattened row-major: one row of inputs per output neuron
    weights = np.concatenate([params[f"{name}.weight"].flatten() for name in LAYERS]).tolist()
    biases = np.concatenate([params[f"{name}.bias"] for name in LAYERS]).tolist()
    print(f"weight std {np.std(weights):.4f}, bias std {np.std(biases):.4f}")

    out = Path(args.out)
    write_inc(out / "nnue_weights.inc", quantize(weights))
    write_inc(out / "nnue_biases.inc", quantize(biases))
    print(f"wrote {len(weights)} weights and {len(biases)} biases to {out}")


if __name__ == "__main__":
    main()
