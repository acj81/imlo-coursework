import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn
import datetime 

# --- DEFINE MODEL ---

class ANDenseBlock(nn.Module):
    def __init__(self, in_channels, conv_layers=4, growth_rate=4, filter_size=3):
        super().__init__()

        # create our convolutional layers based on params given:
        layers = []

        for i in range(conv_layers):
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels + (growth_rate * i), growth_rate, filter_size, padding="same"),
                nn.LazyBatchNorm2d(),
                nn.GELU(),
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
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
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


class ArchimedesNetV23(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=8),
            ANTransBlock(54, 27, 2),
            ANDenseBlock(27, conv_layers=12, growth_rate=16),
            ANTransBlock(219, 110, 4),
            ANDenseBlock(110, conv_layers=18, growth_rate=24),
            ANTransBlock(542, 271, 4),
            ANDenseBlock(271, conv_layers=30, growth_rate=32),
            ANTransBlock(1231, 616, 4),
            ANDenseBlock(616, conv_layers=6, growth_rate=8),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(664),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(2656, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV24(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 6, 1),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=32),
            ANTransBlock(198, 99, 2),
            ANDenseBlock(99, conv_layers=12, growth_rate=16),
            ANTransBlock(291, 146, 4),
            ANDenseBlock(146, conv_layers=18, growth_rate=24),
            ANTransBlock(578, 289, 4),
            ANDenseBlock(289, conv_layers=30, growth_rate=32),
            ANTransBlock(1249, 625, 4),
            ANDenseBlock(625, conv_layers=6, growth_rate=8),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(673),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(2692, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


    def forward(self, x):
        x = self.layers(x)
        return x


class ARDTransLayer(nn.Module):
    def __init__(self, in_channels, img_dim, s=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.LayerNorm(img_dim),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=s, stride=s),
        )


    def forward(self, x):
        x = self.layers(x)
        return x

class ARDFeatureMixer(nn.Module):
    def __init__(self, in_channels, img_dim, growth_rate, filter_size=7):
        super().__init__()

        self.layers = nn.Sequential(
            # need depthwise convolution, so use groups=in_channels to get that in PyTorch
            nn.Conv2d(in_channels, in_channels, kernel_size=filter_size, groups=in_channels, padding="same"),
            nn.LayerNorm(img_dim),
            # use filter_size=channel-dim to mimic a linear layer on 2d images
            nn.Conv2D(in_channels, 4 * in_channels, kerneL_size=1),
            nn.GELU(),
            nn.Conv2D(4 * in_channels, growth_rate, kernel_size=1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ARDStageLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, img_dim, has_trans_layer=True):
        super().__init__()

        # set feature mixers, updating img_dim as we go:
        self.fm_1 = ARDFeatureMixer(in_channels, img_dim, growth_rate)
        img_dim[0] += growth-rate
        
        self.fm_2 = ARDFeatureMixer(in_channels + growth_rate, img_dim, growth_rate)
        img_dim[0] += growth-rate
        
        self.fm_3 = ARDFeatureMixer(in_channels + 2 * growth_rate, img_dim, growth_rate)
        img_dim[0] += growth-rate

        # handle transition layer
        if has_trans_layer:
            self.trans_layer = ARDTransLayer(in_channels + 3 * growth_rate, img_dim, s=1)

        else:
            # if shouldn't have trans layer, set trans layer to return input w/o modification
            self.trans_layer = nn.Identity()       

    def forward(self, x):
        # y is current output, x is current input:
        y = self.fm_1(x)
        x = torch.cat((x, y), 1)
        y = self.fm_2(x)
        x = torch.cat((x, y), 1)
        y = self.fm_3(x)
        x = torch.cat((x, y), 1)
        x = self.trans_layer(x)
        return x


class ARDNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=4, stride=4),
            # Stage 1
            ARDStageLayer(6, 64, (6, 64, 64)),
            ARDStageLayer(198, 64, (198, 64, 64)),
            ARDStageLayer(390, 64, (390, 64, 64), has_trans_layer=False),
            ARDTransLayer(582, img_dim=(582, 64, 64), s=2),
            # Stage 2
            ARDStageLayer(291, 64, (291, 32, 32)),
            ARDStageLayer(483, 64, (483, 32, 32)),
            ARDStageLayer(675, 64, (675, 32, 32), has_trans_layer=False),
            ARDTransLayer(867, img_dim=(867, 32, 32), s=2),
            # Stage 3
            ARDStageLayer(433, 64, (433, 16, 16)),
            ARDStageLayer(625, 64, (625, 16, 16)),
            ARDStageLayer(817, 64, (817, 16, 16)),
            ARDStageLayer(1009, 64, (1009, 16, 16)),
            ARDStageLayer(1201, 64, (1201, 16, 16)),
            ARDStageLayer(1393, 64, (1393, 16, 16)),
            ARDStageLayer(1585, 64, (1585, 16, 16)),
            ARDStageLayer(1777, 64, (1777, 16, 16)),
            ARDStageLayer(1969, 64, (1969, 16, 16)),
            ARDStageLayer(2161, 64, (2161, 16, 16)),
            ARDStageLayer(2353, 64, (2353, 16, 16)),
            ARDStageLayer(2545, 64, (2545, 16, 16), has_trans_layer=False),
            ARDTransLayer(2545, img_dim=(2545, 16, 16), s=2),
            # Stage 4
            ARDStageLayer(1272, 64, (1272, 8, 8)),
            ARDStageLayer(1464, 64, (1464, 8, 8)),
            ARDStageLayer(1656, 64, (1656, 8, 8), has_trans_layer=False),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.LayerNorm(),
            nn.Linear(1656, 37)
        )



# handle accelerators i.e. GPU - if one available, should use that:
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using accelerator: {device}")


model = ARDNet().to(device)


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

learn_rate = 1e-4

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
