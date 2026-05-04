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
    def __init__(self, in_channels, out_channels, s=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=s, stride=s),
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
            ANTransBlock(156, 78, 2),
            ANDenseBlock(78, conv_layers=12, growth_rate=24),
            ANTransBlock(366, 183, 2),
            ANDenseBlock(183, conv_layers=18, growth_rate=24),
            ANTransBlock(615, 307, 2),
            ANDenseBlock(307, conv_layers=30, growth_rate=24),
            ANTransBlock(1027, 514, 2),
            ANDenseBlock(514, conv_layers=6, growth_rate=24),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(658),
            nn.AdaptiveAvgPool2d((1,1)),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(658, 2632),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2632, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x



class ArchimedesNetV14(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 7, 2, stride=2),
            # dense-trans block combos:
            ANDenseBlock(7, conv_layers=4, growth_rate=3),
            ANTransBlock(19, 9, 2),
            ANDenseBlock(9, conv_layers=8, growth_rate=8),
            ANTransBlock(73, 36, 2),
            ANDenseBlock(36, conv_layers=20, growth_rate=13),
            ANTransBlock(296, 148, 2),
            ANDenseBlock(148, conv_layers=21, growth_rate=50),
            ANTransBlock(1198, 599, 2),
            ANDenseBlock(599, conv_layers=64, growth_rate=64),
            ANTransBlock(4695, 2347, 2),
            ANDenseBlock(2347, conv_layers=130, growth_rate=130),
            ANTransBlock(19247, 9623, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(9623),
            nn.AdaptiveAvgPool2d((1,1)),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(9623, 38492),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(38492, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV15(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # convolution to extract features
            nn.Conv2d(3, 38, 4, stride=4),
            # dense-trans block combos:
            ANDenseBlock(38, conv_layers=8, growth_rate=13),
            ANTransBlock(142, 71, 2),
            ANDenseBlock(71, conv_layers=9, growth_rate=24),
            ANTransBlock(287, 143, 2),
            ANDenseBlock(143, conv_layers=32, growth_rate=32),
            ANTransBlock(1167, 583, 2),
            ANDenseBlock(583, conv_layers=65, growth_rate=66),
            ANTransBlock(4873, 2436, 2),
            # final pooling layer to reduce down, batch norm:
            nn.BatchNorm2d(2436),
            nn.AdaptiveAvgPool2d((1,1)),
            # finally, linear classification:
            nn.Flatten(),
            nn.Linear(2436, 9744),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(9744, 37)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV16(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # no stem, to see if it changes anything:
            ANDenseBlock(3, conv_layers=9, growth_rate=64),
            ANTransBlock(579, 289, s=4),
            ANDenseBlock(289, conv_layers=9, growth_rate=104),
            ANTransBlock(1225, 612, s=4),
            ANDenseBlock(612, conv_layers=36, growth_rate=128),
            ANTransBlock(5220, 2610, s=4),
            ANDenseBlock(2610, conv_layers=9, growth_rate=224),
            ANTransBlock(4626, 2313, s=4),
            # final batch norm:
            nn.BatchNorm2d(2313),
            # fully-connected linear classifier at end:
            nn.Flatten(),
            nn.Linear(2313, 9252),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(9252, 37),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ArchimedesNetV17(nn.Module):
    def __init__(self):
        super().__init__()
 
        # define our actual architecture:
        self.layers = nn.Sequential(
            # no stem, to see if it changes anything:
            ANDenseBlock(3, conv_layers=9, growth_rate=64),
            ANTransBlock(579, 289, s=4),
            ANDenseBlock(289, conv_layers=9, growth_rate=128),
            ANTransBlock(1441, 720, s=4),
            ANDenseBlock(720, conv_layers=63, growth_rate=128),
            ANTransBlock(8784, 4392, s=4),
            ANDenseBlock(4392, conv_layers=18, growth_rate=240),
            ANTransBlock(8712, 4356, s=4),
            # final batch norm:
            nn.BatchNorm2d(4356),
            # fully-connected linear classifier at end:
            nn.Flatten(),
            nn.Linear(4356, 17424),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(17424, 37),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_channels, embed_dim, num_heads, num_layers, num_classes, dropout=0.0):
        ''' 
            image_size: height & width of image as single int (assumes height == width, i.e. a square image)
            patch_size: height & width of path as single int (patches are square i.e. height == width)
            num_channels: number of input channels in image (1 for greyscale, 3 for RGB)
            embed_dim: size of the embedding as single int
            num_heads: number of classification heads the transformer should have
            num_layers: number of layers the transformer should have
            num_classes: number of output classes the images have
            
        '''
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch Embedding Layer
        self.patch_embed = nn.Linear(num_channels * patch_size ** 2, embed_dim)
        
        # Learnable Positional Embedding and CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == W == self.patch_size * int(self.num_patches**0.5), "Image size must be divisible by patch size"
        
        # Split image into patches and flatten
        # x shape: [B, C, H, W] -> [B, num_patches, patch_size*patch_size*C]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(B, C, -1, self.patch_size**2)
        x = x.permute(0, 2, 1, 3).reshape(B, -1, C * self.patch_size**2)
        
        # Project to embedding dimension
        x = self.patch_embed(x)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply Transformer
        x = self.transformer(x)
        
        # Use CLS token for classification
        x = x[:, 0]
        x = self.norm(x)
        x = self.head(x)
        return x   


# handle accelerators i.e. GPU - if one available, should use that:
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using accelerator: {device}")


model = ArchimedesNetV16().to(device)

# lot of params for ViT so specify here:

'''

model = VisionTransformer(
    image_size = 256, 
    patch_size = 16, 
    num_channels = 3, 
    embed_dim = 1024, 
    num_heads = 16, 
    num_layers = 24, 
    num_classes = 37, 
    dropout = 0.2,
).to(device)

'''

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

batch_size =16 

epochs = 30

loss_fn = nn.CrossEntropyLoss()

'''
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learn_rate,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
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

        # every few epochs, test our model:

        if (epoch >= 1):
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
