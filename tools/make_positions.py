"""Generate a reproducible set of test positions by random playouts.

Usage: python tools/make_positions.py COUNT [SEED] > positions.txt

Each line is 64 board chars followed by the side to move, in the engine's input format.
The set mixes openings, midgames and endgames, and includes positions where the side to
move must pass.
"""

import random
import sys

import othello


def random_game(rng, plies):
    """Play random moves for up to `plies` plies; also return any positions seen that force a pass."""
    board, player, passes = othello.START, "0", []
    for _ in range(plies):
        moves = othello.legal_moves(board, player)
        if not moves:
            if othello.game_over(board):
                break
            passes.append(board + player)
            player = othello.other(player)
            continue
        board = othello.play(board, player, rng.choice(moves))
        player = othello.other(player)
    return board, player, passes


def random_positions(count, seed, pass_count=4):
    rng = random.Random(seed)
    positions, passes = [], []
    while len(positions) < count - pass_count:
        board, player, _ = random_game(rng, rng.randint(4, 58))
        if not othello.game_over(board):
            positions.append(board + player)
    # forced passes are rare in random play, so search full games for them
    while len(passes) < pass_count:
        passes += random_game(rng, 80)[2]
    return positions + passes[:pass_count]


if __name__ == "__main__":
    count = int(sys.argv[1])
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    print("\n".join(random_positions(count, seed)))
