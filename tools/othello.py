"""Minimal reference Othello rules, shared by the tooling (fixtures, matches, tests).

Positions use the engine's text format: 64 chars, row-major from the top-left,
'.' empty, '0' black, '1' white. Black ('0') moves first.
"""

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
START = "." * 27 + "10" + "." * 6 + "01" + "." * 27


def flips(board, player, row, col):
    """Squares flipped if `player` ('0' or '1') plays at (row, col); empty if illegal."""
    if board[row * 8 + col] != ".":
        return []
    opponent = "1" if player == "0" else "0"
    result = []
    for dr, dc in DIRECTIONS:
        line = []
        r, c = row + dr, col + dc
        while 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == opponent:
            line.append(r * 8 + c)
            r, c = r + dr, c + dc
        if line and 0 <= r < 8 and 0 <= c < 8 and board[r * 8 + c] == player:
            result.extend(line)
    return result


def legal_moves(board, player):
    return [(r, c) for r in range(8) for c in range(8) if flips(board, player, r, c)]


def play(board, player, move):
    """Return the board after `player` plays `move` ((row, col) or None for a pass)."""
    if move is None:
        return board
    row, col = move
    cells = list(board)
    for index in flips(board, player, row, col) + [row * 8 + col]:
        cells[index] = player
    return "".join(cells)


def other(player):
    return "1" if player == "0" else "0"


def game_over(board):
    return not legal_moves(board, "0") and not legal_moves(board, "1")


def score(board):
    return board.count("0"), board.count("1")


def move_to_str(move):
    """(row, col) -> 'd3' style (column letter, 1-based row), None -> 'pass'."""
    if move is None:
        return "pass"
    row, col = move
    return f"{chr(ord('a') + col)}{row + 1}"


def str_to_move(text):
    text = text.strip()
    if text == "pass":
        return None
    return int(text[1:]) - 1, ord(text[0]) - ord("a")


def perft(board, player, depth):
    """Leaf count at `depth` plies; a pass counts as a ply, finished games are leaves."""
    if depth == 0:
        return 1
    moves = legal_moves(board, player)
    if not moves:
        if not legal_moves(board, other(player)):
            return 1
        return perft(board, other(player), depth - 1)
    return sum(perft(play(board, player, m), other(player), depth - 1) for m in moves)
