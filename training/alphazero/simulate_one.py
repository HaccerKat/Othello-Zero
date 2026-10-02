import random
from nn_init import NeuralNetwork, load_model
from mcts import mcts
from board import Board
import time
import numpy as np

nn_name_control = input("Enter the control's model name: ")
nn_name_experimental = input("Enter the experimental's model name: ")
control_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_control + '.pth')
experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_experimental + '.pth')
control_model.eval()
experimental_model.eval()

control_player = random.randint(0, 1)
print("Control Player:", control_player)
current_player = 0
initial_player_board = 0x0000000810000000
initial_opponent_board = 0x0000001008000000
if control_player == 1:
    initial_player_board = 0x0000000810000000
    initial_opponent_board = 0x00000001008000000

# 0 - Black
# 1 - White
start = time.perf_counter()
board = Board(initial_player_board, initial_opponent_board, current_player)
y = 1
while not board.game_ends():
    if current_player == control_player:
        (player_board, opponent_board) = mcts(board, control_model, True, 0, 800, 0.8)
    else:
        (player_board, opponent_board) = mcts(board, experimental_model, True, 0, 800, 0.8)
    policy = board.mcts_policy
    x = np.argmax(policy)
    y *= x
    current_player = 1 - current_player
    board = Board(player_board, opponent_board, current_player)

board.print()
winner = board.get_winner()
current_is_control = current_player == control_player
if winner == 0:
    print("Draw.")
elif (current_is_control and winner == 1) or (not current_is_control and winner == -1):
    print("Control Wins!")
else:
    print("Experimental Wins!")
end = time.perf_counter()
print("Execution Time (s):", end - start)