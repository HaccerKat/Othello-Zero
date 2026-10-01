"""Play the compiled CodinGame bundle against a random opponent through a simulated referee.

Usage: python codingame/test_bundle.py BUNDLE_BINARY [GAMES]

Mimics the referee input the bot expects: player id and board size once, then each turn the
board rows, the opponent's last move(s) (from the second turn on, since the bot enables EXPERT
mode), the legal action count and the actions. When the bot has to pass, the referee skips it
and the opponent's consecutive moves arrive together, comma separated (an assumption about the
referee's format; the bot only relies on each move taking 3 characters). Checks every reply is
legal.
"""

import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import othello  # noqa: E402


# Black's only move is h4; if white answers b8, black must pass while white moves twice.
FORCED_PASS = ("000000..10000000010001010010001.0011011101010111000001110.0000000", "0", ["b8"])


def play(binary, bot, rng, start=(othello.START, "0"), script=()):
    """Play one game; the opponent makes the moves in `script` first, then random ones."""
    script = [othello.str_to_move(m) for m in script]
    proc = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    send = lambda text: (proc.stdin.write(text + "\n"), proc.stdin.flush())
    send(f"{bot}\n8")

    (board, player), first_turn, opponent_moves = start, True, []
    try:
        while not othello.game_over(board):
            legal = othello.legal_moves(board, player)
            if player != bot:
                move = script.pop(0) if script else rng.choice(legal) if legal else None
                opponent_moves.append(othello.move_to_str(move))
                board, player = othello.play(board, player, move), othello.other(player)
                continue
            if not legal:
                # the referee does not ask a player that must pass
                player = othello.other(player)
                continue

            send("\n".join(board[r * 8:r * 8 + 8] for r in range(8)))
            if not first_turn:
                send(",".join(opponent_moves))
            send(str(len(legal)) + "\n" + "\n".join(othello.move_to_str(m) for m in legal))
            reply = proc.stdout.readline().split()
            if len(reply) != 2 or reply[0] != "EXPERT" or othello.str_to_move(reply[1]) not in legal:
                raise AssertionError(f"bad reply {reply} to {board}{player}")
            board, player = othello.play(board, player, othello.str_to_move(reply[1])), othello.other(player)
            first_turn, opponent_moves = False, []
    finally:
        proc.kill()
    black, white = othello.score(board)
    return (black - white) if bot == "0" else (white - black)


if __name__ == "__main__":
    binary = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    rng = random.Random(7)
    for game in range(games):
        bot = "01"[game % 2]
        diff = play(binary, bot, rng)
        print(f"game {game + 1}: bot played {'black' if bot == '0' else 'white'}, disc difference {diff:+d}")
    board, bot, script = FORCED_PASS
    for game in range(3):
        diff = play(binary, bot, rng, (board, bot), script)
        print(f"forced pass game {game + 1}: disc difference {diff:+d}")
    print("all replies legal")
