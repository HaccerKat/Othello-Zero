"""Play a match between two engines that speak the CLI protocol (engine/cli.cpp).

Usage:
  python tools/match.py "CMD_A" "CMD_B" [--openings N] [--plies K] [--seed S]

Each opening is K random plies from the start position, played twice with colours
swapped, so 2N games in total. Prints the score from engine A's point of view.
"""

import argparse
import random
import shlex
import subprocess

import othello


class Engine:
    def __init__(self, command):
        self.proc = subprocess.Popen(shlex.split(command), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

    def move(self, board, player):
        self.proc.stdin.write(board + player + "\n")
        self.proc.stdin.flush()
        return othello.str_to_move(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def random_opening(rng, plies):
    board, player = othello.START, "0"
    for _ in range(plies):
        board = othello.play(board, player, rng.choice(othello.legal_moves(board, player)))
        player = othello.other(player)
    return board, player


def play_game(engines, board, player):
    """engines maps '0'/'1' to Engine. Returns the final board."""
    while not othello.game_over(board):
        legal = othello.legal_moves(board, player)
        move = engines[player].move(board, player)
        if legal and move not in legal:
            raise RuntimeError(f"illegal move {othello.move_to_str(move)} in {board}{player}")
        if not legal:
            move = None
        board = othello.play(board, player, move)
        player = othello.other(player)
    return board


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("engine_a")
    parser.add_argument("engine_b")
    parser.add_argument("--openings", type=int, default=50)
    parser.add_argument("--plies", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    a, b = Engine(args.engine_a), Engine(args.engine_b)
    rng = random.Random(args.seed)
    wins = draws = losses = 0
    disc_diff = 0
    for game in range(args.openings):
        board, player = random_opening(rng, args.plies)
        for a_colour in "01":
            b_colour = othello.other(a_colour)
            final = play_game({a_colour: a, b_colour: b}, board, player)
            black, white = othello.score(final)
            diff = (black - white) if a_colour == "0" else (white - black)
            disc_diff += diff
            wins, draws, losses = wins + (diff > 0), draws + (diff == 0), losses + (diff < 0)
        print(f"opening {game + 1}/{args.openings}: +{wins} ={draws} -{losses}", flush=True)
    a.close()
    b.close()

    games = wins + draws + losses
    print(f"A: {args.engine_a}\nB: {args.engine_b}")
    print(f"A scored {wins + draws / 2}/{games} (+{wins} ={draws} -{losses}), "
          f"average disc difference {disc_diff / games:+.1f}")


if __name__ == "__main__":
    main()
