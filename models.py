import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Net_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp_conv1 = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
                bias = False
            )),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=2))
        ]))
        self.ternary_con2 = nn.Sequential(OrderedDict([
            ("conv1",nn.Conv2d(16, 32, 5, 1, 2, bias=False)),
            ("norm1", nn.BatchNorm2d(32)),
            ("relu1", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(2)),

            ('conv2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False)),
            ('norm2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False)),
            ('norm3', nn.BatchNorm2d(128)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=2, stride=2)),
        ]))

        # fully connected layer, output 10 classes
        self.fp_fc = nn.Linear(1152, 10)  #personalized layer
    def forward(self, x):
        x = self.fp_conv1(x)
        x = self.ternary_con2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.fp_fc(x)
        return output


class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out

class Net_4(nn.Module):
    def __init__(self, n_classes = 10):
        super().__init__()
        self.fp_conv1 = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                bias=False
            )),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=2))
        ]))
        self.ternary_con2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(6,16,5, bias=False)),
            ('relu1', nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(2))
        ]))
        self.fp_fc = nn.Sequential(OrderedDict([
            ("fp_fc1", nn.Linear(16 * 5 * 5, 120)),
            ("fp_fc2", nn.Linear(120, 84)),
            ("fp_fc3", nn.Linear(84, n_classes))
        ]))

    def forward(self, x):
        x = self.fp_conv1(x)
        x = self.ternary_con2(x)
        x = x.view(x.size(0), -1)
        x = self.fp_fc(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        return x

class Net_1(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.name = "Net"
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Net"
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
