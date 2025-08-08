import torch
import board_helper as bh
import time

# Even resnets cannot saturate a mid-tier GPU without inference batching
def batch_inference(boards, model, num_games):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    tensor = bh.bitboards_to_tensor(
        [board.player_board for board in boards],
        [board.opponent_board for board in boards]
    )

    model.eval()
    tensor = tensor.to(device)
    with torch.no_grad():
        policies, values = model(tensor)

    policies = torch.nn.functional.softmax(policies, dim=1)
    policies = policies.detach().cpu().numpy()
    values = values.detach().cpu().numpy()
    for i in range(num_games):
        boards[i].policy_head = policies[i]
        boards[i].value_head = values[i].item()
    end = time.perf_counter()

def mcts(root, model, debug, mode, num_simulations, exploration_constant):
    batch_inference([root], model, 1)
    root.find_next_boards(mode)
    for i in range(num_simulations):
        node = root
        while not node.game_ends():
            # Not fully expanded, so expand
            if not node.next_boards or node.current_child < len(node.next_boards):
                node = node.expand(model)
                break
            # Fully expanded, select child with highest UCT value
            else:
                node = node.select(exploration_constant)

        batch_inference([node], model, 1)
        node.backpropagate(mode)

    result = root.get_next_board(mode)
    if debug:
        root.print()

    return result

def mcts_mp(roots, model, num_games, debug, mode, num_simulations, exploration_constant):
    batch_inference(roots, model, num_games)
    for root in roots:
        root.find_next_boards(mode)
    for _ in range(num_simulations):
        # shallow copy of the roots array
        nodes = roots[:]
        for i in range(num_games):
            while not nodes[i].game_ends():
                # Not fully expanded, so expand
                if not nodes[i].next_boards or nodes[i].current_child < len(nodes[i].next_boards):
                    nodes[i] = nodes[i].expand(model)
                    break
                # Fully expanded, select child with highest UCT value
                else:
                    nodes[i] = nodes[i].select(exploration_constant)

        batch_inference(nodes, model, num_games)
        start = time.perf_counter()
        for i in range(num_games):
            nodes[i].backpropagate(mode)

        end = time.perf_counter()
        globals.state['time_eval_3'] += end - start

    results = [None] * num_games
    for i in range(num_games):
        results[i] = roots[i].get_next_board(mode)
    return results