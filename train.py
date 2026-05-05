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
    def __init__(self, in_channels, out_channels, s=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.MaxPool2d(s),
            nn.BatchNorm2d(out_channels),
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
            nn.Conv2d(3, 12, 1, stride=1),
            # dense-trans block combos:
            ANDenseBlock(12, conv_layers=6, growth_rate=24),
            ANTransBlock(156, 76, 2),
            ANDenseBlock(76, conv_layers=12, growth_rate=24),
            ANTransBlock(364, 182, 4),
            ANDenseBlock(182, conv_layers=18, growth_rate=24),
            ANTransBlock(614, 307, 4),
            ANDenseBlock(307, conv_layers=30, growth_rate=24),
            ANTransBlock(1027, 514, 2),
            ANDenseBlock(514, conv_layers=6, growth_rate=24),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(658),
            nn.AdaptiveAvgPool2d((1,1)),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(658, 512),
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
            nn.Conv2d(3, 12, 1, stride=1),
            # dense-trans block combos:
            ANDenseBlock(12, conv_layers=6, growth_rate=24),
            ANTransBlock(156, 76, 4),
            ANDenseBlock(76, conv_layers=12, growth_rate=24),
            ANTransBlock(364, 182, 4),
            ANDenseBlock(182, conv_layers=18, growth_rate=24),
            ANTransBlock(614, 307, 4),
            ANDenseBlock(307, conv_layers=30, growth_rate=24),
            ANTransBlock(658, 329, 4),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(329),
            nn.AdaptiveAvgPool2d((1,1)),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(329, 1316),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1316, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels, filter_size=3, stride=1):
        super().__init__()
 
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=filter_size, stride=stride, padding="same"),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=filter_size, stride=stride, padding="same"),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )


    def forward(self, x):
        y = self.layers(x)
        return y + x


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()
        
        # architecture here:
        self.layers = nn.Sequential(
            # conv block
            nn.Conv2d(3, 64, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv block 
            nn.Conv2d(64, 128, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # res block
            ResBlock(128),
            # conv block
            nn.Conv2d(128, 256, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # res block
            ResBlock(256),
            # conv block
            nn.Conv2d(256, 512, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # finally, avg pooling and LC
            nn.AdaptiveAvgPool2d((1,1)),
            nn.LazyBatchNorm2d(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 37),
        )


    def forward(self, x):
        # pass through each layer
        x = self.layers(x)
        return x


class ResNet11(nn.Module):
    def __init__(self):
        super().__init__()
        
        # architecture here:
        self.layers = nn.Sequential(
            # conv block
            nn.Conv2d(3, 64, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # conv block 
            nn.Conv2d(64, 128, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # res block
            ResBlock(128),
            # conv block
            nn.Conv2d(128, 256, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # res block
            ResBlock(256),
            # conv block
            nn.Conv2d(256, 512, 3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # res block:
            ResBlock(512),
            # finally, avg pooling and LC
            nn.AdaptiveAvgPool2d((1,1)),
            nn.LazyBatchNorm2d(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(512, 37),
        )

    def forward(self, x):
        # pass through each layer
        x = self.layers(x)
        return x


class ResNet32(nn.Module):
    def __init__(self):
        super().__init__()
        
        # architecture here:
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 7),
            ResBlock(32),
            ResBlock(32),
            nn.Conv2d(32, 64, 2, stride=2),
            ResBlock(64),
            ResBlock(64),
            nn.Conv2d(64, 128, 2, stride=2),
            ResBlock(128),
            ResBlock(128),
            nn.Conv2d(128, 256, 2, stride=2),
            ResBlock(256),
            ResBlock(256),
            nn.Conv2d(256, 512, 2, stride=2),
            ResBlock(512),
            ResBlock(512),
            nn.Conv2d(512, 1024, 2, stride=2),
            ResBlock(1024),
            ResBlock(1024),
            nn.Conv2d(1024, 2048, 2, stride=2),
            ResBlock(2048),
            ResBlock(2048),
            nn.Conv2d(2048, 4096, 2, stride=2),
            ResBlock(4096),
            ResBlock(4096),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(4096, 37),
        )


    def forward(self, x):
        # pass through each layer
        x = self.layers(x)
        return x


# handle accelerators i.e. GPU - if one available, should use that:
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using accelerator: {device}")


model = ResNet9().to(device)


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

batch_size = 64

epochs = 30

loss_fn = nn.CrossEntropyLoss()

'''
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learn_rate,
    momentum = 0.9,
)

'''
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learn_rate,
)

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

        print(f"trained on epoch: {epoch}")

        # every few epochs, test our model:

        if (epoch % 5 == 0):
                # do both test and train datasets to check for overfitting:
                train_accuracy, train_loss = test(
                        model = model,
                        loss_fn = loss_fn,
                        dataloader = train_dataloader,
                        device = device,
                )
                test_accuracy, test_loss = test(
                        model = model,
                        loss_fn = loss_fn,
                        dataloader = test_dataloader,
                        device = device,
                )

                print(f"epoch: {epoch}, test accuracy: {(100*test_accuracy):>0.1f}%, train accuracy: {(100*train_accuracy):>0.1f}%")

# record end time to get idea of speed:
end_time = datetime.datetime.now()

training_time = end_time - start_time

print(f"time to train: {training_time}")

torch.save(model.state_dict(), "model.pth")
