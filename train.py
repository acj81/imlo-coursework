import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn


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
            nn.Conv2d(3, 256, 1, stride=4, bias=False),
            nn.BatchNorm2d(256),
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


# handle accelerators i.e. GPU - if one available, should use that:
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using accelerator: {device}")

model = ResidualAlexNet().to(device)



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
        transforms.Resize([227, 227]),
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
    avg_loss = test_loss / num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")




# --- HYPERPARAM EXPERIMENTATION HERE ---

# hyperparameters:

learn_rate = 0.001

batch_size = 32

epochs = 30

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9)


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

# test, train then test again to show improvement:

test(
    model = model,
    loss_fn = loss_fn,
    dataloader = test_dataloader,
    device = device,
)

for i in range(epochs):
    print(f"epoch: {i}")
    train(
        model = model,
        loss_fn = loss_fn,
        optimizer = optimizer,
        dataloader = train_dataloader,
        device = device,
    )

test(
    model = model,
    loss_fn = loss_fn,
    dataloader = test_dataloader,
    device = device,
)

torch.save(model.state_dict(), "model.pth")
