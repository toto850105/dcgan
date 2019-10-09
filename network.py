import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 0, bias=False),
            nn.MaxPool2d(3, 3),
            nn.Tanh()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 0, bias=False),
            nn.MaxPool2d(2, 2),
            nn.Tanh()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(2*2*64, 200),
            nn.Tanh()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(200, 10),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, 2*2*64)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = x.max().item()
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Sequential(
            # nn.Linear(10, 100),
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
