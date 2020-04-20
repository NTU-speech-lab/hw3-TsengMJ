import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            nn.Dropout(0.3, True),

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.3, True),

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.3, True),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Dropout(0.3, True),
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
            nn.Dropout(0.4, True),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)



class Question2(nn.Module):
    def __init__(self):
        super(Question2, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 20, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            nn.Dropout(0.3, True),

            nn.Conv2d(20, 50, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.3, True),

            nn.Conv2d(50, 100, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.3, True),

        )
        self.fc = nn.Sequential(
            nn.Linear(100*16*16, 500),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(500, 250),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            nn.Linear(250, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)



class Question3(nn.Module):
    def __init__(self):
        super(Question3, self).__init__()
        self.dnn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*128*128, 256),
            nn.ReLU(),
            nn.Dropout(0.3, True),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4, True),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5, True),
            
            nn.Linear(64, 11)
            
        )

    def forward(self, x):
        out = self.dnn(x)

        return out

