import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn
import datetime 

# --- DEFINE MODEL ---

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # convolution and pooling layers - these recognise features within our images:
            nn.Conv2d(3, 96, 11, stride = 4),
            nn.MaxPool2d(3),
            nn.Conv2d(96, 256, 5, padding = 2),
            nn.MaxPool2d(3),
            nn.Conv2d(256, 384, 3, padding = 1),
            nn.Conv2d(384, 384, 3, padding = 1),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.MaxPool2d(3),
            # fully connected layers - these combine those features to categorise the image:
            nn.Flatten(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 37),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits


class ResidualAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage_1 = nn.Sequential(
            # convolution and pooling layers - these recognise features within our images:
            nn.Conv2d(3, 96, 11, stride = 4),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(96, 256, 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )

        self.shortcut_1 = nn.Sequential(
            nn.Conv2d(3, 6, 1, stride=4, bias=False),
            nn.BatchNorm2d(6),
        )

        self.stage_2 = nn.Sequential(
            nn.Conv2d(256, 384, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )

        self.shortcut_2 = nn.Sequential(
            nn.Conv2d(384, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.fc_layers = nn.Sequential(
            # fully connected layers - these combine those features to categorise the image:
            nn.Flatten(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 37),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        res_1 = self.shortcut_1(x)
        x = self.stage_1(x)
        x += res_1
        res_2 = self.shortcut_2(x)
        x = self.stage_2(x)
        x += res_2
        x = self.fc_layers(x)
        return x

class ANV1Block(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size, pool_size):
        super().__init__()
        # building blocks for larger network defined here for reusability:
        self.activ = nn.LeakyReLU()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, filter_size, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, filter_size, padding="same")
        self.pool = nn.MaxPool2d(pool_size)

        self.reduce_channels = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        # reduce channels on input so we can add later:
        res = self.reduce_channels(x)
        # convolutions
        x = self.conv1(x)
        x = self.activ(x)
        x = self.conv2(x)
        # add 1st residual back in for smoother gradient
        x += res
        x = self.activ(x)
        x = self.pool(x)
        return x

class ArchimedesNetV1(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # feature detection using CNNs:
            ANV1Block(3, 32, 5, 2),
            ANV1Block(32, 32, 4, 2),
            ANV1Block(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            ANV1Block(64, 64, 3, 2),
            ANV1Block(64, 128, 3, 2),
            ANV1Block(128, 128, 2, 2),
            ANV1Block(128, 256, 2, 2),
            ANV1Block(256, 256, 1, 2),
            # image classification using features we detected:
            nn.Flatten(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.Dropout(0.3),
            nn.Linear(512, 216),
            nn.Dropout(0.2),
            nn.Linear(216, 37),
        )       

    def forward(self, x):
        x = self.layers(x)
        return x

class ANDenseBlock(nn.Module):
    def __init__(self, in_channels, conv_layers=4, growth_rate=4, filter_size=3):
        super().__init__()

        # create our convolutional layers based on params given:
        layers = []

        for i in range(conv_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels + (growth_rate * i), growth_rate, filter_size, padding="same"),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
            ))

        # convert an array of modules into sequence, so we can call forward on it
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # iterate for each submodule:
        for layer in self.layers:
            y = layer(x)
            x = torch.cat((x, y), dim=1)
        # return final input - input across all the features 
        return x


class ANTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_size)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV2(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 32, 1),
            # dense-trans block combos:
            ANDenseBlock(32),
            ANTransBlock(48, 32, 2),
            ANDenseBlock(32),
            ANTransBlock(48, 16, 2),
            ANDenseBlock(16),
            ANTransBlock(32, 8, 2),
            ANDenseBlock(8),
            ANTransBlock(24, 4, 2),
            # final pooling layer to reduce down, batch norm:
            nn.AvgPool2d(2),
            nn.BatchNorm2d(4),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(256, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV7(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 32, 1),
            # dense-trans block combos:
            ANDenseBlock(32),
            ANTransBlock(48, 40, 2),
            ANDenseBlock(40),
            ANTransBlock(56, 48, 2),
            ANDenseBlock(48),
            ANTransBlock(64, 56, 2),
            ANDenseBlock(56),
            ANTransBlock(72, 64, 2),
            ANDenseBlock(64),
            ANTransBlock(80, 72, 2),
            ANDenseBlock(72),
            ANTransBlock(88, 64, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(64),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV8(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 64, 1),
            # dense-trans block combos:
            ANDenseBlock(64),
            ANTransBlock(80, 68, 2),
            ANDenseBlock(68),
            ANTransBlock(84, 72, 2),
            ANDenseBlock(72),
            ANTransBlock(88, 76, 2),
            ANDenseBlock(76),
            ANTransBlock(92, 80, 2),
            ANDenseBlock(80),
            ANTransBlock(96, 84, 2),
            ANDenseBlock(84),
            ANTransBlock(100, 88, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(88),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(1408, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV9(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 64, 1),
            # dense-trans block combos:
            ANDenseBlock(64, conv_layers=5),
            ANTransBlock(84, 68, 2),
            ANDenseBlock(68, conv_layers=5),
            ANTransBlock(88, 72, 2),
            ANDenseBlock(72, conv_layers=5),
            ANTransBlock(92, 76, 2),
            ANDenseBlock(76, conv_layers=5),
            ANTransBlock(96, 80, 2),
            ANDenseBlock(80, conv_layers=5),
            ANTransBlock(100, 84, 2),
            ANDenseBlock(84, conv_layers=5),
            ANTransBlock(104, 64, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(64),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV10(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 64, 1),
            # dense-trans block combos:
            ANDenseBlock(64, conv_layers=5, growth_rate=5),
            ANTransBlock(89, 69, 2),
            ANDenseBlock(69, conv_layers=5, growth_rate=5),
            ANTransBlock(94, 74, 2),
            ANDenseBlock(74, conv_layers=5, growth_rate=5),
            ANTransBlock(99, 79, 2),
            ANDenseBlock(79, conv_layers=5, growth_rate=5),
            ANTransBlock(104, 84, 2),
            ANDenseBlock(84, conv_layers=5, growth_rate=5),
            ANTransBlock(109, 89, 2),
            ANDenseBlock(89, conv_layers=5, growth_rate=5),
            ANTransBlock(114, 96, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(96),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV11(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=16),
            ANTransBlock(102, 51, 2),
            ANDenseBlock(51, conv_layers=12, growth_rate=16),
            ANTransBlock(243, 121, 4),
            ANDenseBlock(121, conv_layers=18, growth_rate=16),
            ANTransBlock(409, 204, 4),
            ANDenseBlock(204, conv_layers=30, growth_rate=16),
            ANTransBlock(684, 342, 4),
            ANDenseBlock(342, conv_layers=6, growth_rate=16),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(438),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(1752, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV12(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=24),
            ANTransBlock(150, 75, 2),
            ANDenseBlock(75, conv_layers=12, growth_rate=24),
            ANTransBlock(363, 182, 4),
            ANDenseBlock(182, conv_layers=18, growth_rate=24),
            ANTransBlock(614, 307, 4),
            ANDenseBlock(307, conv_layers=30, growth_rate=24),
            ANTransBlock(1027, 514, 4),
            ANDenseBlock(514, conv_layers=6, growth_rate=24),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(658),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(2632, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV13(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=32),
            ANTransBlock(198, 49),
            ANDenseBlock(49, conv_layers=12, growth_rate=32),
            ANTransBlock(433, 109),
            ANDenseBlock(109, conv_layers=30, growth_rate=32),
            ANTransBlock(1069, 267),
            ANDenseBlock(267, conv_layers=6, growth_rate=32),
            ANTransBlock(459, 114),
            # final batch norm, avg pool to reduce features:
            nn.BatchNorm2d(114),
            nn.AvgPool2d(2),
            # linear classifier:
            nn.Flatten(),
            nn.Linear(7296,37),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class ArchimedesNetV14(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=24, filter_size=7),
            ANTransBlock(150, 75, 2),
            ANDenseBlock(75, conv_layers=12, growth_rate=24, filter_size=5),
            ANTransBlock(363, 182, 4),
            ANDenseBlock(182, conv_layers=18, growth_rate=24, filter_size=3),
            ANTransBlock(614, 307, 4),
            ANDenseBlock(307, conv_layers=30, growth_rate=24, filter_size=3),
            ANTransBlock(1027, 514, 4),
            ANDenseBlock(514, conv_layers=6, growth_rate=24, filter_size=3),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(658),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(2632, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV15(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=24, filter_size=3),
            ANTransBlock(150, 75, 2),
            ANDenseBlock(75, conv_layers=12, growth_rate=24, filter_size=3),
            ANTransBlock(363, 182, 4),
            ANDenseBlock(182, conv_layers=18, growth_rate=24, filter_size=5),
            ANTransBlock(614, 307, 4),
            ANDenseBlock(307, conv_layers=30, growth_rate=24, filter_size=7),
            ANTransBlock(1027, 514, 4),
            ANDenseBlock(514, conv_layers=6, growth_rate=24, filter_size=3),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(658),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(2632, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV16(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=24),
            ANTransBlock(150, 75, 2),
            ANDenseBlock(75, conv_layers=12, growth_rate=24),
            ANTransBlock(363, 182, 2),
            ANDenseBlock(182, conv_layers=12, growth_rate=24),
            ANTransBlock(470, 235, 2),
            ANDenseBlock(235, conv_layers=18, growth_rate=24),
            ANTransBlock(667, 333, 2),
            ANDenseBlock(333, conv_layers=18, growth_rate=24),
            ANTransBlock(765, 382, 2),
            ANDenseBlock(382, conv_layers=30, growth_rate=24),
            ANTransBlock(1102, 551, 2),
            ANDenseBlock(551, conv_layers=12, growth_rate=24),
            ANTransBlock(839, 420, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(420),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(1680, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV17(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=24),
            ANTransBlock(150, 75, 2),
            ANDenseBlock(75, conv_layers=12, growth_rate=24),
            ANTransBlock(363, 182, 2),
            ANDenseBlock(182, conv_layers=12, growth_rate=24),
            ANTransBlock(470, 235, 2),
            ANDenseBlock(235, conv_layers=18, growth_rate=24),
            ANTransBlock(667, 333, 2),
            ANDenseBlock(333, conv_layers=18, growth_rate=24),
            ANTransBlock(765, 382, 2),
            ANDenseBlock(382, conv_layers=30, growth_rate=24),
            ANTransBlock(1102, 551, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(551),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(8816, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV18(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture here:
        self.final_pool = nn.AvgPool2d(2)
        self.pool8 = nn.MaxPool2d(8)
        
        self.conv1 = nn.Conv2d(3, 6, 1)

        self.dense_1 = nn.Sequential(
            ANDenseBlock(3, growth_rate=8),
            ANTransBlock(35, 18),
            ANDenseBlock(18, growth_rate=8),
            ANTransBlock(50, 25),
            ANDenseBlock(25, growth_rate=8),
            ANTransBlock(57, 29),
            ANDenseBlock(29, growth_rate=8),
        )


        self.dense_2 = nn.Sequential(
            ANDenseBlock(64, growth_rate=8),
            ANTransBlock(96, 48),
            ANDenseBlock(48, growth_rate=8),
            ANTransBlock(80, 40),
            ANDenseBlock(40, growth_rate=8),
            ANTransBlock(72, 36),
            ANDenseBlock(36, growth_rate=8),
        )

        self.lc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(528,512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37),
        )

    def forward(self, x):
        # first dense layer
        y = self.dense_1(x)
        # pool so we can concatenate
        x = self.pool8(x)
        x = torch.cat((y, x), 1)
        # 2nd dense, concatenate again
        y = self.dense_2(x)
        x = self.pool8(x)
        x = torch.cat((y, x), 1)
        # final pool before classification
        x = self.final_pool(x)
        # linear classifier:
        x = self.lc(x)
        return x


class ArchimedesNetV19(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture here:
        self.pool_8 = nn.MaxPool2d(8)
        self.pool_4 = nn.MaxPool2d(4)
        
        self.conv_1 = nn.Conv2d(3, 6, 1)

        self.dense_1 = nn.Sequential(
            ANDenseBlock(6, growth_rate=8),
            ANTransBlock(38, 19),
            ANDenseBlock(19, growth_rate=8),
            ANTransBlock(51, 26),
            ANDenseBlock(26, growth_rate=8),
            ANTransBlock(58, 29),
            ANDenseBlock(29, growth_rate=8),
        )


        self.dense_2 = nn.Sequential(
            ANDenseBlock(67, growth_rate=8),
            ANTransBlock(99, 50),
            ANDenseBlock(50, growth_rate=8),
            ANTransBlock(82, 41),
            ANDenseBlock(41, growth_rate=8),
            ANTransBlock(73, 37),
            ANDenseBlock(37, growth_rate=8),
        )

        self.dense_3 = nn.Sequential(
            ANDenseBlock(136, growth_rate=8),
            ANTransBlock(168, 84),
            ANDenseBlock(84, growth_rate=8),
            ANTransBlock(116, 58),
            ANDenseBlock(58, growth_rate=8),
            ANTransBlock(90, 45, pool_size=1),
            ANDenseBlock(45, growth_rate=8),
        )

        self.lc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(213,37),
        )

    def forward(self, x):
        # first dense layer
        x = self.conv_1(x)
        y = self.dense_1(x)
        # pool so we can concatenate
        x = self.pool_8(x)
        x = torch.cat((y, x), 1)
        # 2nd dense, concatenate again
        y = self.dense_2(x)
        x = self.pool_8(x)
        x = torch.cat((y, x), 1)
        # 3rd dense, concatenate again:
        y = self.dense_3(x)
        x = self.pool_4(x)
        x = torch.cat((y, x), 1)
        # linear classifier:
        x = self.lc(x)
        return x


class ArchimedesNetV20(nn.Module):
    def __init__(self):
        super().__init__()
        # define our actual architecture here:

        self.pool_2 = nn.MaxPool2d(2)

        self.final_pool = nn.AvgPool2d(4)

        self.dense_1 = nn.Sequential(
            ANDenseBlock(3, growth_rate=8),
            ANTransBlock(35, 18, pool_size=1),
            ANDenseBlock(18, growth_rate=8),
            ANTransBlock(50, 25, pool_size=1),
            ANDenseBlock(25, growth_rate=8),
            ANTransBlock(57, 29, pool_size=1),
            ANDenseBlock(29, growth_rate=8),
            ANTransBlock(61, 31, pool_size=2),
        )
        self.dense_2 = nn.Sequential(
            ANDenseBlock(34, growth_rate=8),
            ANTransBlock(66, 33, pool_size=1),
            ANDenseBlock(33, growth_rate=8),
            ANTransBlock(65, 32, pool_size=1),
            ANDenseBlock(32, growth_rate=8),
            ANTransBlock(64, 32, pool_size=1),
            ANDenseBlock(32, growth_rate=8),
            ANTransBlock(64, 32, pool_size=2),
        )
        self.dense_3 = nn.Sequential(
            ANDenseBlock(66, growth_rate=8),
            ANTransBlock(98, 49, pool_size=1),
            ANDenseBlock(49, growth_rate=8),
            ANTransBlock(81, 41, pool_size=1),
            ANDenseBlock(41, growth_rate=8),
            ANTransBlock(73, 37, pool_size=1),
            ANDenseBlock(37, growth_rate=8),
            ANTransBlock(69, 35, pool_size=2),
        )
        self.dense_4 = nn.Sequential(
            ANDenseBlock(101, growth_rate=8),
            ANTransBlock(133, 66, pool_size=1),
            ANDenseBlock(66, growth_rate=8),
            ANTransBlock(98, 49, pool_size=1),
            ANDenseBlock(49, growth_rate=8),
            ANTransBlock(81, 41, pool_size=1),
            ANDenseBlock(41, growth_rate=8),
            ANTransBlock(73, 37, pool_size=2),
        )

        self.lc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 37),
        )

    def forward(self, x):
        # dense layers:
        y = self.dense_1(x)
        x = self.pool_2(x)
        x = torch.cat((x, y), 1)
        y = self.dense_2(x)
        x = self.pool_2(x)
        x = torch.cat((x, y), 1)
        y = self.dense_3(x)
        x = self.pool_2(x)
        x = torch.cat((x, y), 1)
        y = self.dense_4(x)
        x = self.pool_2(x)
        x = torch.cat((x, y), 1)
        # linear classifier:
        x = self.final_pool(x)
        x = self.lc(x)
        return x


# handle accelerators i.e. GPU - if one available, should use that:
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using accelerator: {device}")

model = ArchimedesNetV20().to(device)


# --- DEFINE OUR TRAIN, TEST AND DATA AUGMENTATION FUNCTIONS ---

# specify function to convert from integer label (0-36 inclusive) to corresponding binary array:
to_one_hot = transforms.Compose(
        [
            transforms.Lambda(lambda x : torch.zeros(37, dtype=torch.float).scatter_(dim=0, index=torch.tensor(x).long(), value=1)),
            #transforms.ToTensor(),
        ]
)

# function to resize images, convert to tensor:
image_transform = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ]
)


# -- train, test loops:

def train(dataloader, model, loss_fn, optimizer, device):
    # model into training mode - good practice
    model.train()
    # size of dataset, use for computing accuracy:
    size = len(dataloader.dataset)
    # train on each batch
    for batch, (features, labels) in enumerate(dataloader):
        # move both to device:
        features = features.to(device)
        labels = labels.to(device)
        # prediction and loss
        pred = model(features)
        loss = loss_fn(pred, labels)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn, device):
    # set model to evaluation mode - good practice
    model.eval()
    # size, number of batches:
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # reduces resource usage
    with torch.no_grad():
        # test on each batch
        for batch, (features, labels) in enumerate(dataloader):
            # move both to device:
            features = features.to(device)
            labels = labels.to(device)
            # get accuracy, loss for batch
            pred = model(features)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()

        
    # calculate loss, accuracy as percentages
    avg_loss = test_loss / size
    accuracy = correct / size
    return accuracy, avg_loss



# --- HYPERPARAM EXPERIMENTATION HERE ---

# hyperparameters:

learn_rate = 0.0001

batch_size = 16

epochs = 30

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


# specify test, train datasets:
train_data = datasets.OxfordIIITPet(
    root="data",
    split="trainval",
    download=True,
    transform=image_transform,
    target_transform=to_one_hot,
)

test_data = datasets.OxfordIIITPet(
    root="data",
    split="test",
    download=True,
    transform=image_transform,
    target_transform=to_one_hot,
)

# dataloaders for those datasets:

train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True,
)

test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
)


# --- TRAIN AND TEST HERE ---

start_time = datetime.datetime.now()

# train our model:
for epoch in range(1, epochs + 1):
        # train for this epoch
        train(
            model = model,
            loss_fn = loss_fn,
            optimizer = optimizer,
            dataloader = train_dataloader,
            device = device,
        )

        # every 5 epochs, test our model:

        if (epoch % 5 == 0):
                accuracy, loss = test(
                        model = model,
                        loss_fn = loss_fn,
                        dataloader = test_dataloader,
                        device = device,
                )

                print(f"epoch: {epoch}, accuracy: {(100*accuracy):>0.1f}%")

# record end time to get idea of speed:
end_time = datetime.datetime.now()

training_time = end_time - start_time

print(f"time to train: {training_time}")

torch.save(model.state_dict(), "model.pth")
