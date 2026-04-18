import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch import nn


# --- DEFINE MODEL AND LOAD WEIGHTS ---

class ArchimedesNet(nn.Module):
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
            nn.ReLU(),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using accelerator: {device}")

model = ArchimedesNet().to(device)
model.load_state_dict(torch.load("model.pth", weights_only=True))


# --- TEST THE MODEL ---

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

# call our test function:

loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 32

test_data = datasets.OxfordIIITPet(
    root="data",
    split="test",
    download=True,
    transform=image_transform,
    target_transform=to_one_hot,
)

test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
)

test(
    model = model,
    loss_fn = loss_fn,
    dataloader = test_dataloader,
    device = device,
)

