from torch import nn
import torch
import copy
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DeepNet(nn.Module):
    def __init__(self, dim_x, dim_y, hidden=None):
        super(DeepNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden[1], dim_y)
        )
        # Sigmoid is necessary to get smooth score

    def forward(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y

    def forward_and_get_fea(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y, fea


class DeepNet_Large(nn.Module):
    def __init__(self, dim_x, dim_y):
        super(DeepNet_Large, self).__init__()

        hidden = [1024, 512, 256, 64]

        self.net = nn.Sequential(
            nn.Linear(dim_x, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], hidden[2]),
            nn.ReLU(),
            nn.Linear(hidden[2], hidden[3]),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden[3], dim_y)
        )
        # Sigmoid is necessary to get smooth score

    def forward(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y

    def forward_and_get_fea(self, x):
        fea = self.net(x)
        y = self.fc(fea)

        y = nn.functional.sigmoid(y)
        
        return y, fea

class LinearClassifier(nn.Module):
    def __init__(self, dim_x, dim_y):
        super(LinearClassifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(dim_x, dim_y)
        )
        # Sigmoid is necessary to get smooth score

    def forward(self, x):
        y = self.fc(x)

        y = nn.functional.sigmoid(y)
        
        return y
