import torch
from torch import nn
import torch.nn.functional as F
class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, labels_uint8, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.labels_uint8 = labels_uint8
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        X = torch.tensor(self.inputs_uint8[idx], dtype=torch.uint8)
        y = torch.tensor(self.labels_uint8[idx], dtype=torch.uint8)
        X = X.float()
        y = y.float() / 128

        if self.transform:
            X = self.transform(X)

        return X, y

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(129, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class NeuralNetworkNNUE(torch.nn.Module):
    # Need to experiment to find a better NN structure
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(128, 256)
        self.layer2 = torch.nn.Linear(256, 32)
        self.layer3 = torch.nn.Linear(32, 32)
        self.value = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        value = F.tanh(self.value(x))
        return value