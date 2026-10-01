"""Bundle the engine into a single source file for CodinGame.

Usage: python codingame/build_bundle.py [OUTPUT]   (default: build/codingame_bundle.cpp)

CodinGame takes one source file of at most 100k characters. This concatenates the engine
sources and codingame/main.cpp, and swaps the compiled-in weight arrays for the base-93
text encoding from tools/weights_codec.py, decoded at startup.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from weights_codec import encode, read_inc, to_c_string  # noqa: E402

SOURCES = [
    "engine/nnue.h", "engine/board.h", "engine/search.h",
    "engine/nnue.cpp", "engine/board.cpp", "engine/search.cpp",
    "codingame/main.cpp",
]
SIZE_LIMIT = 100_000

WEIGHTS_BLOCK = re.compile(r"// BEGIN WEIGHTS.*?// END WEIGHTS\n", re.S)
DECODER = """\
int16_t QUANT_WEIGHTS[WEIGHTS_SZ], QUANT_BIASES[BIASES_SZ];

// Weights are embedded as text to fit the size limit, see tools/weights_codec.py
static void decode_weights(const char* text, int16_t* out, int count) {
    for (int i = 0; i < count; i++) {
        char c = *text++;
        int x = c - '!' - 46;
        if (c == '~') {
            x = (*text++ - '!' - 46) * 93;
            x += *text++ - '!';
        }
        out[i] = x;
    }
}

static const char* weights_string = WEIGHTS;
static const char* biases_string = BIASES;
static const bool weights_loaded = (decode_weights(weights_string, QUANT_WEIGHTS, WEIGHTS_SZ),
                                    decode_weights(biases_string, QUANT_BIASES, BIASES_SZ), true);
"""


def bundle():
    system_includes, bodies = [], []
    for name in SOURCES:
        text = (ROOT / name).read_text()
        if name == "engine/nnue.cpp":
            weights = to_c_string(encode(read_inc(ROOT / "engine/weights/nnue_weights.inc")))
            biases = to_c_string(encode(read_inc(ROOT / "engine/weights/nnue_biases.inc")))
            decoder = DECODER.replace("WEIGHTS;", weights + ";").replace("BIASES;", biases + ";")
            text, found = WEIGHTS_BLOCK.subn(lambda _: decoder, text)
            assert found == 1, "weights block not found in engine/nnue.cpp"
        lines = []
        for line in text.splitlines():
            if line.startswith("#include <"):
                if line not in system_includes:
                    system_includes.append(line)
            elif not (line.startswith('#include "') or line == "#pragma once"):
                lines.append(line)
        bodies.append(f"// ---- {name} ----\n" + "\n".join(lines).strip() + "\n")
    header = "// Othello-Zero engine, bundled for CodinGame by codingame/build_bundle.py\n"
    return header + '#pragma GCC optimize("O3")\n' + "\n".join(system_includes) + "\n\n" + "\n".join(bodies)


if __name__ == "__main__":
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "codingame_bundle.cpp"
    source = bundle()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source)
    print(f"wrote {output} ({len(source):,} / {SIZE_LIMIT:,} characters)")
    if len(source) > SIZE_LIMIT:
        sys.exit("bundle exceeds CodinGame's size limit")
