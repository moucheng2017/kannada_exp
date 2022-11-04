import torch.nn as nn


# Define a network:
def conv_layer(dim_in, dim_out, k=3, p=1, s=1):
    layers = nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=k, padding=p, stride=s),
        nn.BatchNorm2d(dim_out),
        nn.LeakyReLU(0.1)
    )
    return layers


def linear_layer(dim_in, dim_out):
    layers = nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.BatchNorm1d(dim_out),
        nn.LeakyReLU(0.1)
    )
    return layers


class Vgglight(nn.Module):
    def __init__(self):
        super(Vgglight, self).__init__()
        self.conv1 = conv_layer(1, 64)
        self.conv2 = conv_layer(64, 64)
        #         self.conv3 = conv_layer(64, 64)
        self.pooling = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.4)

        self.conv4 = conv_layer(64, 128)
        self.conv5 = conv_layer(128, 128)
        #         self.conv6 = conv_layer(128, 128)
        self.dropout2 = nn.Dropout(0.4)

        self.conv7 = conv_layer(128, 256)
        self.conv8 = conv_layer(256, 256)
        #         self.conv9 = conv_layer(256, 512)
        #         self.conv10 = conv_layer(512, 512)
        self.dropout3 = nn.Dropout(0.4)

        self.flatten = nn.Flatten(start_dim=1)
        #         self.linear = linear_layer(8192, 256) # when use all of layers
        self.linear = linear_layer(4096, 256)
        self.dropout4 = nn.Dropout(0.1)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #         x = self.conv3(x)
        x = self.pooling(x)
        x = self.dropout1(x)

        x = self.conv4(x)
        x = self.conv5(x)
        #         x = self.conv6(x)
        x = self.pooling(x)
        x = self.dropout2(x)

        x = self.conv7(x)
        x = self.conv8(x)
        #         x = self.conv9(x)
        #         x = self.conv10(x)
        x = self.pooling(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        #         print(x.size())
        x = self.linear(x)
        x = self.dropout4(x)
        x = self.output(x)

        return x