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
    def __init__(self, in_channels):
        super().__init__()

        # common activation function across all layers:
        self.activ = nn.ReLU()

        # declare conv layers here:
        self.conv1 = nn.Conv2d(in_channels, in_channels, 5, padding="same")
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 1, padding="same")
        self.conv3 = nn.Conv2d(in_channels * 3, in_channels, 5, padding="same")
        self.conv4 = nn.Conv2d(in_channels * 4, in_channels, 1, padding="same")
        self.conv5 = nn.Conv2d(in_channels * 3, in_channels, 5, padding="same")
        self.conv6 = nn.Conv2d(in_channels * 4, in_channels, 1, padding="same")

    def forward(self, x):
        # y is running output, x is running input:
        y = self.activ(self.conv1(x))
        x = torch.concat((y, x), 1)
        y = self.activ(self.conv2(x))
        x = torch.concat((y, x), 1)
        y = self.activ(self.conv3(x))
        x = torch.concat((y, x), 1)
        y = self.activ(self.conv4(x))
        x = torch.concat((y, x), 1)
        y = self.activ(self.conv5(x))
        x = torch.concat((y, x), 1)
        y = self.activ(self.conv5(x))
        x = torch.concat((y, x), 1)
        return y


class ANTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.MaxPool2d(pool_size),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV2(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 1),
            ANDenseBlock(64),
            ANTransBlock(64, 128, 2),
            ANDenseBlock(128),
            ANTransBlock(128, 64, 2),
            ANDenseBlock(64),
            ANTransBlock(64, 32, 2),
            ANDenseBlock(32),
            ANTransBlock(32, 16, 2),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.Dropout(0.1),
            nn.Linear(512, 37),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# handle accelerators i.e. GPU - if one available, should use that:
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using accelerator: {device}")

model = ArchimedesNetV2().to(device)


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

learn_rate = 0.001

batch_size = 32

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
