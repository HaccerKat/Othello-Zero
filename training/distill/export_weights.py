import torch
from nn_init import NeuralNetworkNNUE
import numpy as np
def load_model(model_class, checkpoint_path, device=None, **model_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

name = input("Enter NNUE Name: ")
model = load_model(NeuralNetworkNNUE, './models/model_weights_' + name + '.pth')

state_dict = model.state_dict()

params = {k: v.cpu().numpy() for k, v in state_dict.items()}

open('./models/weights_' + name + '.txt', 'w').close()
open('./models/biases_' + name + '.txt', 'w').close()
all_weights = []
all_biases = []
for x in range(4):
    weights = params[('layer' + str(x + 1) if x < 3 else 'value') + '.weight']
    weights = weights.flatten()
    weights = weights.tolist()
    file1 = open('./models/weights_' + name + '.txt', "a+")
    file1.write(" ".join(str(x) for x in weights))
    file1.write('\n')
    file1.close()
    all_weights += weights

    biases = params[('layer' + str(x + 1) if x < 3 else 'value') + '.bias']
    biases = biases.tolist()
    file2 = open('./models/biases_' + name + '.txt', "a+")
    file2.write(" ".join(str(x) for x in biases))
    file2.write('\n')
    file2.close()
    all_biases += biases

# Use ascii 33 (!) to 125 (})
# ascii 126 (~) is used as a seperator for larger values
# Prints 2 \ so it can be hardcoded
# Cannot be used in simulate_games.cpp anymore
BOUND = 0.05
QUANT_MULT = 46.5 / BOUND
def f(x):
    x = round(x * QUANT_MULT)
    if -46 <= x <= 46:
        c = chr(ord('!') + x + 46)
        return c if c != '\\' and c != '\'' and c != '\"' else '\\' + c
    d1 = x // 93
    d2 = x % 93
    c1 = chr(ord('!') + d1 + 46)
    c2 = chr(ord('!') + d2)
    return '~' + (c1 if c1 != '\\' and c1 != '\'' and c1 != '\"' else '\\' + c1) + (c2 if c2 != '\\' and c2 != '\'' and c2 != '\"' else '\\' + c2)

print(np.std(all_weights))
print(np.std(all_biases))
all_weights = list(map(f, all_weights))
all_biases = list(map(f, all_biases))
with open('./models/weights_' + name + '_quantized.txt', 'w') as f:
    f.write(''.join(str(x) for x in all_weights))
    f.write('\n')
with open('./models/biases_' + name + '_quantized.txt', 'w') as f:
    f.write(''.join(str(x) for x in all_biases))
    f.write('\n')