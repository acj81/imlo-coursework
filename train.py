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
            nn.Conv2d(3, 6, 4, stride=2),
            # dense-trans block combos:
            ANDenseBlock(6, conv_layers=6, growth_rate=24),
            ANTransBlock(150, 75, 2),
            ANDenseBlock(75, conv_layers=12, growth_rate=24),
            ANTransBlock(363, 182, 2),
            ANDenseBlock(182, conv_layers=18, growth_rate=24),
            ANTransBlock(614, 307, 2),
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
            nn.Conv2d(in_channels, out_channels= in_channels // 2, kernel_size=s, stride=s),
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
            nn.Conv2d(in_channels, 4 * in_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(4 * in_channels, growth_rate, kernel_size=1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class ARDStageLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, img_dim, has_trans_layer=True):
        super().__init__()

        # set feature mixers, updating img_dim as we go:
        self.fm_1 = ARDFeatureMixer(in_channels, img_dim, growth_rate)
        img_dim[0] += growth_rate
        
        self.fm_2 = ARDFeatureMixer(in_channels + growth_rate, img_dim, growth_rate)
        img_dim[0] += growth_rate
        
        self.fm_3 = ARDFeatureMixer(in_channels + (2 * growth_rate), img_dim, growth_rate)
        img_dim[0] += growth_rate

        # handle transition layer
        if has_trans_layer:
            self.trans_layer = ARDTransLayer(in_channels + (3 * growth_rate), img_dim, s=1)

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
            ARDStageLayer(6, img_dim=[6, 64, 64], growth_rate=128),
            ARDStageLayer(195, img_dim=[195, 64, 64], growth_rate=128),
            ARDStageLayer(289, img_dim=[289, 64, 64], growth_rate=128, has_trans_layer=False),
            ARDTransLayer(673, img_dim=[673, 64, 64], s=2),
            # Stage 2
            ARDStageLayer(336, img_dim=[336, 32, 32], growth_rate=192),
            ARDStageLayer(456, img_dim=[456, 32, 32], growth_rate=192),
            ARDStageLayer(516, img_dim=[516, 32, 32], growth_rate=192, has_trans_layer=False),
            ARDTransLayer(1092, img_dim=[1092, 32, 32], s=2),
            # Stage 3
            ARDStageLayer(546, img_dim=[546, 16, 16], growth_rate=256),
            ARDStageLayer(657, img_dim=[657, 16, 16], growth_rate=256),
            ARDStageLayer(712, img_dim=[712, 16, 16], growth_rate=256),
            ARDStageLayer(740, img_dim=[740, 16, 16], growth_rate=256),
            ARDStageLayer(754, img_dim=[754, 16, 16], growth_rate=256),
            ARDStageLayer(761, img_dim=[761, 16, 16], growth_rate=256),
            ARDStageLayer(764, img_dim=[764, 16, 16], growth_rate=256),
            ARDStageLayer(766, img_dim=[766, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256),
            ARDStageLayer(767, img_dim=[767, 16, 16], growth_rate=256, has_trans_layer=False),
            ARDTransLayer(1535, img_dim=[1535, 16, 16], s=2),
            # Stage 4
            ARDStageLayer(767, img_dim=[767, 8, 8], growth_rate=360),
            ARDStageLayer(923, img_dim=[923, 8, 8], growth_rate=360),
            ARDStageLayer(1001, img_dim=[1001, 8, 8], growth_rate=360, has_trans_layer=False),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.LayerNorm((2081, 1, 1)),
            nn.Flatten(),
            nn.Linear(2081, 37),
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


#model = ArchimedesNetV12().to(device)

# lot of params for ViT so specify here:


model = VisionTransformer(
    image_size = 256, 
    patch_size = 16, 
    num_channels = 3, 
    embed_dim = 768, 
    num_heads = 12, 
    num_layers = 12, 
    num_classes = 37, 
    dropout = 0.0,
).to(device)




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

learn_rate = 1e-3

batch_size = 16

epochs = 30

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learn_rate,
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
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

        if (epoch % 1 == 0):
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
