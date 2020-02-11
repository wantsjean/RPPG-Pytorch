import torch
from torch import nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, bias=False, bn=None, conv_mode='2D'):
        super(BasicBlock,self).__init__()

        assert conv_mode in ['1D','2D']
        if conv_mode == '2D':
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias)
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=bias)
        
        self.bn = bn

        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class RPPG(nn.Module):
    def __init__(self, losses = ['ce', 'mse'], num_classes=128, drop_rate=0.25):
        super(RPPG,self).__init__()
        self.s1 = BasicBlock(1,  16, (5, 11), 1, 0, bn=nn.BatchNorm2d(16), conv_mode='2D')
        self.s2 = BasicBlock(16, 16, (5, 11), 1, 0, bn=nn.BatchNorm2d(16), conv_mode='2D')
        self.s3 = BasicBlock(16, 16, (5, 11), 1, 0, bn=nn.BatchNorm2d(16), conv_mode='2D')
        self.s4 = BasicBlock(16, 16, (5, 11), 1, 0, bn=nn.BatchNorm2d(16), conv_mode='2D')
        self.s5 = BasicBlock(16, 16, (2, 11), 1, 0, bn=nn.BatchNorm2d(16), conv_mode='2D')

        self.losses = losses
        self.drop = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)

        if 'ce' in losses:
            self.fc_ce1 = nn.Linear(16*14, 128)
            self.fc_ce2 = nn.Linear(128, num_classes)
        if 'mse' in losses:
            self.fc_mse1 = nn.Linear(16*14,128)
            self.fc_mse2 = nn.Linear(128,32)
            self.fc_mse3 = nn.Linear(32,1)

    def forward(self,x):
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x) 
        x = x.view(x.size(0),-1)
        out = {}

        if 'ce' in self.losses:
            out['ce'] = self.fc_ce2(self.relu(self.drop(self.fc_ce1(x))))
        if 'mse' in self.losses:
            out['mse'] = self.fc_mse3(self.relu(self.fc_mse2(self.relu(self.drop(self.fc_mse1(x))))))
        
        return out

if __name__ == "__main__":
    model = RPPG()
    inputs = torch.randn((8,1,18,64))
    print(model(inputs))