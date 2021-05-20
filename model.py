import torch
import torch.nn as nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class transpose_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transpose_block, self).__init__()

        self.trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, x_cat):
        x = self.trans(x)

        x = torch.cat((x_cat, x), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.down_conv1 = conv_block(in_channels, 64)
        self.down_conv2 = conv_block(64, 128)
        self.down_conv3 = conv_block(128, 256)
        self.down_conv4 = conv_block(256, 512)
        self.down_conv5 = conv_block(512, 1024)

        self.max_pool = nn.MaxPool2d(2)

        self.up_conv1 = transpose_block(1024, 512)
        self.up_conv2 = transpose_block(512, 256)
        self.up_conv3 = transpose_block(256, 128)
        self.up_conv4 = transpose_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.down_conv1(x)  # 64 channels

        x2 = self.down_conv2(self.max_pool(x1))  # 128

        x3 = self.down_conv3(self.max_pool(x2))  # 256

        x4 = self.down_conv4(self.max_pool(x3))  # 512

        x5 = self.down_conv5(self.max_pool(x4))  # 1024

        x = self.up_conv1(x5, x4)  # 512
        x = self.up_conv2(x, x3)  # 256
        x = self.up_conv3(x, x2)  # 128
        x = self.up_conv4(x, x1)  # 64

        x = self.final_conv(x)  # out_channels

        return x


if __name__ == "__main__":
    inp = torch.rand((1, 3, 512, 512))
    model = UNet(in_channels=3, out_channels=1)
    output = model(inp)
    print(output.shape)
