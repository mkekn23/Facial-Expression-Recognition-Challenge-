import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_Shallow(nn.Module):
    """architecture 1: baseline MLP (1 hidden layer)"""
    def __init__(self):
        super(MLP_Shallow, self).__init__()
        self.fc1 = nn.Linear(2304, 128)
        self.fc2 = nn.Linear(128, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP_Deep(nn.Module):
    """architecture 2: deeper MLP (2 hidden layer)"""
    def __init__(self):
        super(MLP_Deep, self).__init__()
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP_Regularized(nn.Module):
    """
    architecture 3: deep MLP with Dropout and regularization
    """
    def __init__(self):
        super(MLP_Regularized, self).__init__()
        self.fc1 = nn.Linear(2304, 512)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 7)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x
  
class SimpleCNN(nn.Module):
    """
    architecture 4: Simple Convolutional Neural Network (Baseline CNN)
    2 convulational and 2 FC layer.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
     
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) #conv 1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # maxpooling
        
        # in: 16, out: 32 filter
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) #conv 2
        
        # after MaxPool 48x48 image -> 12x12 (48 -> 24 -> 12)
        self.fc1 = nn.Linear(32 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7) # 7 emotions

    def forward(self, x):
        # [batch_size, 1, 48, 48]
        x = self.pool(F.relu(self.conv1(x)))  #  [batch_size, 16, 24, 24]
        x = self.pool(F.relu(self.conv2(x)))  #  [batch_size, 32, 12, 12]
        
        # for kast layers: [batch_size, 32*12*12]
        x = x.view(-1, 32 * 12 * 12)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepCNN(nn.Module):
    """
    architecture 5: deeper CNN with regularization (BatchNorm and Dropout)
    3 Conv ბლოკი + classificator
    """
    def __init__(self):
        super(DeepCNN, self).__init__()
        
        # 48x48 -> 24x24
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        #  24x24 -> 12x12
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        #  12x12 -> 6x6
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # dropout
        self.feature_dropout = nn.Dropout2d(0.25)
        
        # classificator (now image is 6x6  and we have 128 filters)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc_dropout = nn.Dropout(0.5) # 50% dropout
        self.fc2 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.feature_dropout(x)
        
        # flatten
        x = x.view(-1, 128 * 6 * 6)
        
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x

  
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # skip connection — if channels are different, we use 1x1 convulational for sizes correction
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)  # skip connection!

class ResNetFER(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer1 = ResidualBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2)         # 24x24
        self.layer2 = ResidualBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)         # 12x12
        self.layer3 = ResidualBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)         # 6x6
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # GlobalAvgPool
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 7)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        return self.head(x)
