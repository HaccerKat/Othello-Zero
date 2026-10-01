import torch
from torch import nn
import torch.nn.functional as F
    
class BottleneckBlock(nn.Module):
    def __init__(self, channels):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Skip connection
        return F.relu(out)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.input_conv = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)

        self.res_blocks = nn.Sequential(
            *[BottleneckBlock(64) for _ in range(20)]
        )

        self.policy_conv = nn.Conv2d(64, 8, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(8)
        self.policy_fc = nn.Linear(8 * 8 * 8, 64)

        self.value_conv = nn.Conv2d(64, 4, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.input_conv(x)))

        x = self.res_blocks(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, policy, value, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.policy = policy
        self.value = value
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        inputs = self.inputs_uint8[idx].detach().float()
        policy = self.policy[idx].detach()
        value = self.value[idx].detach()
        if self.transform:
            inputs = self.transform(inputs)

        return inputs, policy, value

class NeuralNetworkNNUE(torch.nn.Module):
    # Need to experiment to find a better NN structure
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(128, 384)
        self.layer2 = torch.nn.Linear(384, 32)
        self.layer3 = torch.nn.Linear(32, 16)
        self.value = torch.nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        value = F.tanh(self.value(x))
        return value

class DatasetNNUE(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, value, transform = None):
        self.inputs_uint8 = inputs_uint8
        self.value = value
        self.transform = transform

    def __len__(self):
        return len(self.inputs_uint8)

    def __getitem__(self, idx):
        inputs = torch.tensor(self.inputs_uint8[idx], dtype=torch.uint8).float()
        value = torch.tensor(self.value[idx], dtype=torch.float32)
        if self.transform:
            inputs = self.transform(inputs)

        return inputs, value

def load_model(model_class, checkpoint_path, device=None, **model_kwargs):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model