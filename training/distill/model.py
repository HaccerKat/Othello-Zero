import torch
import torch.nn.functional as F


class NeuralNetworkNNUE(torch.nn.Module):
    """The distilled value network that the engine runs (engine/nnue.h): 128 -> 256 -> 32 -> 32 -> 1."""

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


class DatasetNNUE(torch.utils.data.Dataset):
    def __init__(self, inputs_uint8, value, transform=None):
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
