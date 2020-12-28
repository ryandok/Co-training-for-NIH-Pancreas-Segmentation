import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):
    def __init__(self):
        super(VNet, self).__init__()

    def forward(self, *input):
        pass


class EncoderBlock(nn.Module):
    def __init__(self, n_conv, n_filters_in, n_filters_out, normalization='none'):
        super(EncoderBlock, self).__init__()

        ops = []
        for i in range(n_conv):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=3, stride=1, padding=1))

            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_conv-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x) + x
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()

    def forward(self, *input):
        pass


if __name__ == '__main__':
    """
    test arc
    """
    conv = EncoderBlock(1, 1, 16, normalization='batchnorm')
    input = torch.randn((1, 1, 128, 128, 64))
    conv_output = conv(input)
    print('d')
