from .swinYNet_part import *

class swinYNet(nn.Module):
  def __init__(self, cls, seg, in_channels=1, num_classes=3, grid=7, residual=False):
    super(swinYNet, self).__init__()
    self.cls = cls
    self.seg = seg
    self.in_channels = in_channels
    self.num_classes = num_classes
    self.grid = grid
    self.residual = residual
    self.encoder = Encoder(self.in_channels)
    if self.seg:
      self.decoder = Decoder(self.num_classes, self.residual)
    if self.cls:
      self.head = Head(self.grid, self.num_classes)

  def forward(self, x):
    x1, x2, x3, feature_map = self.encoder(x)
    if self.cls and not self.seg:
      x = self.head(feature_map)
      return x
    if self.seg and not self.cls:
      x = self.decoder(x1, x2, x3, feature_map)
      return x
    if self.seg and self.cls:
      x_seg = self.decoder(x1, x2, x3, feature_map)
      x_cls = self.head(feature_map)
      return x_cls, x_seg

if __name__ == '__main__':

  net = swinYNet(cls=True, seg=True, in_channels=1, num_classes=3, grid=1, residual=False)
  image = torch.rand(1, 1, 224, 224)
  x_cls, x_seg = net(image)
  print(x_cls.size(), x_seg.size())
