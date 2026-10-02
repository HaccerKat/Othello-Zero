import random
import time
import math
import torch
import torch.multiprocessing as mp
import os

from mcts import mcts_mp
from board import Board
from nn_init import NeuralNetwork, load_model
from multiprocessing_helper import execute_gpu

def simulate_games(parameters):
    identifier, control_model, experimental_model, num_simulations, num_games, exploration_constant = parameters
    control_player = random.randint(0, 1)
    current_player = 0
    initial_player_board = 0x0000000810000000
    initial_opponent_board = 0x0000001008000000
    if control_player == 1:
        initial_player_board = 0x0000000810000000
        initial_opponent_board = 0x00000001008000000
    boards = [Board(initial_player_board, initial_opponent_board, current_player) for _ in range(num_games)]
    draws, control_wins, experimental_wins = 0, 0, 0
    move_num = 0
    while boards:
        if current_player == control_player:
            new_boards = mcts_mp(boards, control_model, len(boards), False, 1, num_simulations, exploration_constant)
        else:
            new_boards = mcts_mp(boards, experimental_model, len(boards), False, 1, num_simulations, exploration_constant)

        current_player = 1 - current_player
        boards = []
        for player_board, opponent_board in new_boards:
            board = Board(player_board, opponent_board, current_player)
            current_is_control = current_player == control_player
            if board.game_ends():
                winner = board.get_winner()
                if winner == 0:
                    draws += 1
                elif (current_is_control and winner == 1) or (not current_is_control and winner == -1):
                    control_wins += 1
                else:
                    experimental_wins += 1
            else:
                boards.append(board)

        move_num += 1
        print("At move number:", move_num)
    return draws, control_wins, experimental_wins

# 0 - Black
# 1 - White
def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    torch.set_num_threads(1)

    nn_name_control = input("Enter the control's model name: ")
    nn_name_experimental = input("Enter the experimental's model name: ")
    control_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_control + '.pth')
    experimental_model = load_model(NeuralNetwork, 'models/model_weights_' + nn_name_experimental + '.pth')
    control_model.eval()
    experimental_model.eval()

    start = time.perf_counter()
    current_game = 0

    num_games_to_simulate = int(input("Enter the number of games: "))
    inference_batch_size = 32
    num_simulations = 800
    exploration_constant = 0.8
    draws, control_wins, experimental_wins = 0, 0, 0
    jobs = [(i, control_model, experimental_model, num_simulations, inference_batch_size, exploration_constant) for i in range(num_games_to_simulate // inference_batch_size)]

    print("Simulating Games...")
    games = execute_gpu(simulate_games, jobs)

    for result_draws, result_control_wins, result_experimental_wins in games:
        draws += result_draws
        control_wins += result_control_wins
        experimental_wins += result_experimental_wins
        current_game += 1

    avg_score = (1.0 * experimental_wins + 0.5 * draws) / num_games_to_simulate
    error = 1.96 * math.sqrt(avg_score * (1 - avg_score) / num_games_to_simulate)
    lower_bound = 100.0 * (avg_score - error)
    upper_bound = 100.0 * (avg_score + error)
    print("Number of Draws:", draws)
    print("Number of Control Wins:", control_wins)
    print("Number of Experimental Wins:", experimental_wins)
    print("Experimental WR: " + str(avg_score * 100.0) + "%")
    print("95% Confidence Interval: [" + str(lower_bound) + "%, " + str(upper_bound) + "%]")
    end = time.perf_counter()
    print("Execution time (s):", end - start)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()