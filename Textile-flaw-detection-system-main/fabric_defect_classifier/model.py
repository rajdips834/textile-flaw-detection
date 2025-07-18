import torch.nn as nn

class FabricDefectModel(nn.Module):
    def __init__(self):
        super(FabricDefectModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Assume 2 classes: good / defect
        )

    def forward(self, x):
        return self.network(x)
