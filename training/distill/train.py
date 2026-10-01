import numpy as np
import torch
from torch.utils.data import DataLoader

from nn_init import NeuralNetworkNNUE, DatasetNNUE
def load_128bit_samples(filename):
    with open(filename, 'rb') as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)

    all_bits = np.unpackbits(raw)
    num_samples = all_bits.size // 128

    inputs = all_bits.reshape((num_samples, 128))
    return inputs

def load_values(filename):
    with open(filename, 'r') as f:
        values = torch.tensor(list(map(float, f.readline().split())), dtype=torch.float32)
    values = values.reshape(-1, 1)
    return values

inputs = load_128bit_samples("./datasets/features.bin")
values = load_values("./datasets/values.txt")
dataset = DatasetNNUE(inputs, values)
training_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

BATCH_SIZE = 64
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

learning_rate = 0.0004
model = NeuralNetworkNNUE()
loss_fn = torch.nn.MSELoss()

# CPU training since this is a small model
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    split_size = size // BATCH_SIZE // 20
    for batch, (input, value) in enumerate(dataloader):
        prediction = model(input)
        loss = loss_fn(prediction, value)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch + 1) % split_size == 0:
            loss, current = loss.item(), batch * BATCH_SIZE
            print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for input, value in dataloader:
            prediction = model(input)
            test_loss += loss_fn(prediction, value).item()

    test_loss /= num_batches
    print(f"Avg Loss: {test_loss:>8f} \n")
    return test_loss

patience = 2
epochs_without_improvement = 0
best_val_loss = float("inf")
epochs = 300
bestNN = None
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    if t >= 4:
        learning_rate = min(0.0002, learning_rate)
    if t >= 8:
        learning_rate = min(0.0001, learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4, momentum=0.9)
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    torch.save(model.state_dict(), 'models_nnue/model_weights_' + str(t + 1) + '.pth')
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        bestNN = model
        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1
        if learning_rate > 0.00005:
            learning_rate /= 2
            model = bestNN
        if epochs_without_improvement >= patience:
            break  # early stopping

torch.save(bestNN.state_dict(), 'models_nnue/best.pth')