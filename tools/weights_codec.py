"""Text codec for the quantized NNUE weights.

CodinGame accepts a single source file of limited size, so the int16 weights are
embedded as a string using printable ASCII ('!' .. '~', 94 chars):

  * a value in [-46, 46] is one char: chr(value + 46 + 33)
  * anything else is '~' followed by two chars q, r with value = q * 93 + r,
    where q is offset like a single value and r is in [0, 92]

The decoder lives in C++ (see codingame/build_bundle.py); this module mirrors it.

Usage:
  python tools/weights_codec.py extract codingame/submission.cpp engine/weights
      decode the weight strings embedded in a submission into .inc files
"""

import re
import sys
from pathlib import Path

OFFSET = 33   # '!'
SMALL = 46    # single-char range is [-SMALL, SMALL]
BASE = 93
ESCAPE = "~"


def encode(values):
    out = []
    for value in values:
        if -SMALL <= value <= SMALL:
            out.append(chr(value + SMALL + OFFSET))
        else:
            q, r = divmod(value, BASE)
            out.append(ESCAPE + chr(q + SMALL + OFFSET) + chr(r + OFFSET))
    return "".join(out)


def decode(text, count):
    values, i = [], 0
    while len(values) < count:
        c = text[i]
        i += 1
        value = ord(c) - OFFSET - SMALL
        if c == ESCAPE:
            value = (ord(text[i]) - OFFSET - SMALL) * BASE + (ord(text[i + 1]) - OFFSET)
            i += 2
        values.append(value)
    if i != len(text):
        raise ValueError(f"{len(text) - i} trailing chars after {count} values")
    return values


def to_c_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def from_c_string(literal):
    return re.sub(r"\\(.)", r"\1", literal)


def write_inc(path, values, per_line=32):
    lines = [", ".join(map(str, values[i:i + per_line])) + "," for i in range(0, len(values), per_line)]
    Path(path).write_text("\n".join(lines) + "\n")


def read_inc(path):
    return [int(v) for v in Path(path).read_text().replace(",", " ").split()]


def extract(submission, out_dir, weights_count=42016, biases_count=321):
    source = Path(submission).read_text()
    strings = dict(re.findall(r'std::string (\w+)_string = "((?:[^"\\]|\\.)*)";', source))
    out = Path(out_dir)
    write_inc(out / "nnue_weights.inc", decode(from_c_string(strings["weights"]), weights_count))
    write_inc(out / "nnue_biases.inc", decode(from_c_string(strings["biases"]), biases_count))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
