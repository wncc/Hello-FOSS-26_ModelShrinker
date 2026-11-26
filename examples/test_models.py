import torch
import torch.nn.functional as F
from torch import nn


class SmallMLP(nn.Module):
    """Plain MLP."""

    def __init__(self, in_features: int = 784, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallVGG(nn.Module):
    """VGG-style CNN: Conv -> BN -> ReLU blocks and a Linear head."""

    def __init__(self, num_classes: int = 10, cfg=(32, 32, "M", 64, 64, "M", 128, "M")):
        super().__init__()
        layers, in_ch = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(in_ch, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class SmallResNet(nn.Module):
    """ResNet with shortcut connections."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.layer1 = BasicBlock(16, 16)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        return self.fc(torch.flatten(self.pool(x), 1))


class SmallMobileNet(nn.Module):
    """MobileNet-style, with depthwise convs."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(in_ch, out_ch, stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
            )

        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU6(inplace=True))
        self.blocks = nn.Sequential(block(16, 32, 1), block(32, 64, 2), block(64, 64, 1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class SmallFunctionalCNN(nn.Module):
    """forward() written with F.relu / x.view, convs with bias and no BatchNorm."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = torch.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (2, 2))
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), 0.3, training=self.training)
        return self.fc2(x)


class SmallTabularMLP(nn.Module):
    """Tabular MLP with BatchNorm1d and Dropout."""

    def __init__(self, in_features: int = 20, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class SmallLSTM(nn.Module):
    """2-layer LSTM over (batch, time, features); classifies from the last time step."""

    def __init__(self, in_features: int = 16, hidden: int = 32, num_classes: int = 10):
        super().__init__()
        self.rnn = nn.LSTM(in_features, hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


class SmallGRU(nn.Module):
    """Bidirectional GRU, averaged over time."""

    def __init__(self, in_features: int = 16, hidden: int = 24, num_classes: int = 5):
        super().__init__()
        self.rnn = nn.GRU(in_features, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden, num_classes)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out.mean(dim=1))


class SmallTransformer(nn.Module):
    """nn.TransformerEncoder (2 layers, 4 heads) over (batch, time, features), mean-pooled."""

    def __init__(self, in_features: int = 16, d_model: int = 32, num_classes: int = 10):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.head(self.encoder(self.proj(x)).mean(dim=1))


TEST_MODELS = {
    "SmallMLP": (SmallMLP, (2, 1, 28, 28)),
    "SmallVGG": (SmallVGG, (2, 3, 32, 32)),
    "SmallResNet": (SmallResNet, (2, 3, 32, 32)),
    "SmallMobileNet": (SmallMobileNet, (2, 3, 32, 32)),
    "SmallFunctionalCNN": (SmallFunctionalCNN, (1, 3, 24, 40)),  # batch of 1, non-square
    "SmallTabularMLP": (SmallTabularMLP, (4, 20)),
    "SmallLSTM": (SmallLSTM, (2, 7, 16)),  # (batch, time, features)
    "SmallGRU": (SmallGRU, (3, 5, 16)),
    "SmallTransformer": (SmallTransformer, (2, 6, 16)),
}
