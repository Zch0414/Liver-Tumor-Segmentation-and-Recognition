
import torch 
from torchvision.models import swin_t 
from torch import nn
import torch.nn.functional as F


class Encoder(nn.Module):
  def __init__(self, in_channels=1):
    super(Encoder, self).__init__()
    self.in_channels = in_channels
    self.stage_1 = swin_t(weights='DEFAULT').features[0: 2] # linear projection and 2 stbs
    self.stage_1[0][0] = nn.Conv2d(self.in_channels, 96, kernel_size=(4, 4), stride=(4, 4))
    self.stage_2 = swin_t(weights='DEFAULT').features[2: 4] # patch merging and 2 stbs
    self.stage_3 = swin_t(weights='DEFAULT').features[4: 6] # patch merging and 6 stbs
    self.stage_4 = swin_t(weights='DEFAULT').features[6: 8] # patch merging and 2 stbs
    self.norm = swin_t(weights='DEFAULT').norm
  def forward(self, x):
    x1 = self.stage_1(x)
    x2 = self.stage_2(x1)
    x3 = self.stage_3(x2)
    feature_map = self.norm(self.stage_4(x3))
    return x1.permute(0, 3, 1, 2), x2.permute(0, 3, 1, 2), x3.permute(0, 3, 1, 2), feature_map.permute(0, 3, 1, 2)

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1):
    super(DoubleConv, self).__init__()
    if not mid_channels:
      mid_channels = out_channels
    self.double_conv = nn.Sequential(
      nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=padding),
      nn.BatchNorm2d(mid_channels),
      nn.GELU(approximate='none'),
      nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
      nn.BatchNorm2d(out_channels)
      # nn.GELU(approximate='none')
    )
  def forward(self, x):
    x = self.double_conv(x)
    return x

class Up(nn.Module):
  def __init__(self, in_channels, out_channels, shortcut=False):
    super(Up, self).__init__()
    self.shortcut = shortcut
    self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
    self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

  def forward(self, x_pre, x):
    x_now = self.up(x)
    x = torch.cat([x_pre, x_now], dim=1)
    x = self.conv(x)
    if not self.shortcut:
      x = F.gelu(x, approximate='none')
    return x

class Decoder(nn.Module):
  def __init__(self, num_classes=3, residual=False):
    super(Decoder, self).__init__()
    self.num_classes = num_classes
    self.residual = residual
    self.up_1 = Up(768, 384)
    self.up_2 = Up(384, 192)
    self.up_3 = Up(192, 48, shortcut=self.residual)
    self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    if self.residual:
      self.up_shortcut = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
      self.shortcut = nn.Conv2d(768, 48, kernel_size=1)
      self.conv_residual = DoubleConv(48, 48)
      self.conv = DoubleConv(48, self.num_classes, 24)
    else:
      self.conv = nn.Conv2d(48, self.num_classes, kernel_size=1)
  def forward(self, x1, x2, x3, feature_map):
    x = self.up_1(x3, feature_map)
    x = self.up_2(x2, x)
    x = self.up_3(x1, x)
    if self.residual:
      y = self.up_shortcut(feature_map)
      y = self.shortcut(y)
      x = F.gelu(x + y, approximate='none')
      x = F.gelu(self.conv_residual(x), approximate='none')
    x = self.up_4(x)
    x = self.conv(x)
    return x

class Head(nn.Module):
  def __init__(self, grid=7, num_classes=3):
    super(Head, self).__init__()
    self.num_classes = num_classes
    if grid == 3:
      self.head = nn.Conv2d(768, self.num_classes, kernel_size=3, stride=2, padding=0)
    elif grid == 5:
      self.head = nn.Conv2d(768, self.num_classes, kernel_size=3, stride=1, padding=0)
    elif grid == 7:
      self.head = nn.Conv2d(768, self.num_classes, kernel_size=1)
    else:
      self.head = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(768, 3)
      )
  def forward(self, x):
    x = self.head(x)
    return x
