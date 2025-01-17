import torch.nn as nn
import math

## CNN with dropout
class AlexNet3D_Dropout(nn.Module):

    def __init__(self, num_classes=2, p=.5):
        super(AlexNet3D_Dropout, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Sequential(nn.Dropout(p),
                                        nn.Linear(128, 64),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p),
                                        nn.Linear(64, num_classes),
                                        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.classifier(x)
        return [x, xp]

#### note to self - took below code from original AA code
class AlexNet3D_Dropout_Regression(nn.Module):

    def __init__(self, num_classes = 1, p=0.5):
        super(AlexNet3D_Dropout_Regression, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),

            nn.Conv3d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU(inplace=True),

            nn.Conv3d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm3d(192),
            nn.ReLU( inplace=True),

            nn.Conv3d(192, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU( inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=3),
        )
       
        self.regressor = nn.Sequential(nn.Dropout(p),
                                        #nn.Linear(1536, 64), ADNI
                                        nn.Linear(256, 64),
                                        nn.ReLU( inplace=True),
                                        nn.Dropout(p),
                                        nn.Linear(64, num_classes),
                                       )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
          
    def forward(self,x):
        xp = self.features(x)
        x = xp.view(xp.size(0), -1)
        x = self.regressor(x).squeeze()
        return [x]
