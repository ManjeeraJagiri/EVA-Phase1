import torch.nn as nn
import torch.nn.functional as F

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(0.01),

        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(0.01),     

        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(0.01), 
    ) 

    self.transblock1 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, padding=0, stride=2, bias=False),
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(0.01),

       
    ) 

    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout(0.01),

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout(0.01),     

        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Dropout(0.01), 
    ) 

    self.transblock2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, padding=0, stride=2, bias=False), 
    )

    self.convblock3 = nn.Sequential(
        depthwise_separable_conv(32, 64), 
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(0.01),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0, dilation=2, bias=False), 
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.Dropout(0.01),     
    ) 

    self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1, padding=0, bias=False), 

            nn.AvgPool2d(4),
    )

  def forward(self, x):
      x = self.convblock1(x)
      x = self.transblock1(x)
      x = self.convblock2(x)
      x = self.transblock2(x)
      x = self.convblock3(x)
      # x = self.transblock3(x)
      x = self.convblock4(x)
      x = x.reshape(-1, 10)
      return F.log_softmax(x, dim=-1)