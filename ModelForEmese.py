import os
import matplotlib.pyplot as plt
import cv2 
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import albumentations as A
from albumentations.pytorch import ToTensorV2

image_size_full = 768 #256

class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, target_size=(image_size_full, image_size_full), augment = True):
        self.images_path = images_path
        self.masks_path = masks_path
        self.image_files = sorted(os.listdir(images_path))
        self.mask_files = sorted(os.listdir(masks_path))
        self.target_size = target_size 

        self.augment = augment

        # self.augmentations = A.Compose([
        #     A.HorizontalFlip(p=0.5),
        #     A.Rotate(limit=15),
        #     A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     A.RandomBrightnessContrast(p=0.2),
        #     A.RandomResizedCrop(height=target_size[0], width=target_size[1], scale=(0.8, 1.0)),
        #     A.GaussianBlur(blur_limit=3, p=0.3),
        #     A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        #     A.CLAHE(clip_limit=2, p=0.3),
        #     A.Perspective(scale=(0.05, 0.1), p=0.3),
        #     A.GridDistortion(p=0.3),
        #     A.PadIfNeeded(min_height=target_size[0], min_width=target_size[1], border_mode=cv2.BORDER_REFLECT_101, p=0.3),
        # ], additional_targets={'mask': 'image'})

        self.augmentations = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.GaussianBlur(blur_limit=1, p=0.1),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.1),
            A.CLAHE(clip_limit=1, p=0.1),
        ], additional_targets={'mask': 'image'})




       
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_path, self.image_files[idx])
        mask_path = os.path.join(self.masks_path, self.mask_files[idx])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)    

        if self.target_size is not None:
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)

        if self.augment:
            augmented = self.augmentations(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
            
        image = image.astype(np.float32) / 255.0
        
        mask = self.transform_mask(mask)

        coords_intensities, labels = self.generate_all_coordinates_intensities_and_labels(image, mask)

        return coords_intensities, labels


    def transform_mask(self, mask):
        mask_transformed = np.zeros_like(mask)
        mask_transformed[mask == 0] = 0
        mask_transformed[mask == 128] = 1
        mask_transformed[mask == 255] = 2

        return mask_transformed

    def generate_all_coordinates_intensities_and_labels(self, image, mask):
        H, W = image.shape
        
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        coords = np.stack((y_coords, x_coords), axis=-1).reshape(-1, 2)
        
        coords = coords.astype(np.float32) / np.array([H, W], dtype=np.float32)
        
        intensities = image.flatten()
        labels = mask.flatten()
        
        coords_intensities = np.hstack([coords, intensities.reshape(-1, 1)])
        
        coords_intensities = torch.tensor(coords_intensities, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return coords_intensities, labels

from torch.utils.data import DataLoader

images_path = 'Werkstudent/RAVIR Dataset/train/training_images' ## to mody
masks_path = 'Werkstudent/RAVIR Dataset/train/training_masks' ## to modify
batch_size = 4   ## to modify based on GPU

dataset = SegmentationDataset(images_path, masks_path, target_size=(image_size_full, image_size_full))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

coords_intensities, sampled_labels = next(iter(dataloader))
print(coords_intensities.shape, sampled_labels.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs

    def forward(self, x):
        frequencies = torch.linspace(0, self.num_freqs - 1, self.num_freqs, device=x.device)
        frequencies = 2.0 ** frequencies
        x = x.unsqueeze(-1) * frequencies.unsqueeze(0).unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x.view(x.shape[0], -1)
    
class AdaptiveDropout(nn.Module):
    def __init__(self, initial_p=0.5, decay_factor=0.95):
        super(AdaptiveDropout, self).__init__()
        self.p = initial_p
        self.decay_factor = decay_factor

    def forward(self, x):
        if self.training:
            return F.dropout(x, p=self.p, training=True)
        else:
            return x

    def step(self):
        self.p *= self.decay_factor 
    
# Sine Layer
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=15.0):
        super(SineLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0

    def forward(self, x):
        x = self.linear(x)
        x = torch.sin(self.omega_0 * x)
        return x

class INRSegmentationModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=image_size_full, num_layers=5, num_freqs=10, initial_dropout_p=0.5):
        super(INRSegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.pos_enc = PositionalEncoding(num_freqs)
        self.mlp = nn.ModuleList()
        input_dim = num_freqs * 2 * 2 + 1  

        self.dropouts = nn.ModuleList([AdaptiveDropout(initial_dropout_p) for _ in range(num_layers - 1)])

        self.mlp.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            SineLayer(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim*hidden_dim),
            self.dropouts[0]
        ))

        for i in range(1, num_layers - 2):
            self.mlp.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim*hidden_dim),
                self.dropouts[i]
            ))
        self.mlp.append(nn.Linear(hidden_dim, num_classes))

    def forward(self, coords_intensities):
        bs, ns, _ = coords_intensities.size()
        coords, intensities = coords_intensities[:, :, :-1], coords_intensities[:, :, -1].view((bs, ns,1))
        x = self.pos_enc(coords).view((bs, ns, -1))
        x = torch.cat([x, intensities], dim=-1)
        for layer in self.mlp[:-1]:
            x = layer(x)
        x = self.mlp[-1](x)
        return x 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

import torch.nn.functional as F

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convert class indices to one-hot encoding
        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Flatten the inputs and targets for each class
        inputs = inputs.view(-1, self.num_classes)
        targets_one_hot = targets_one_hot.view(-1, self.num_classes)
        
        # Compute the intersection and union for each class
        intersection = (inputs * targets_one_hot).sum(dim=0)
        union = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        # Compute the Dice coefficient for each class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Average Dice coefficient across classes
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.optim as optim

num_classes = 3 
num_layers = 6
hidden_dim = image_size_full
model = INRSegmentationModel(num_classes=num_classes, hidden_dim=hidden_dim, num_layers=num_layers).to(device)


optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0001)

criterion = MultiClassDiceLoss(num_classes) # nn.CrossEntropyLoss()

scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30, verbose=True)


num_epochs = 3000

losses = []


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0 
    for batch in dataloader:
        coords_intensities, labels = batch
        coords_intensities, labels = coords_intensities.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(coords_intensities)
        loss = criterion(outputs.view((-1, num_classes)), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        

    epoch_loss /= len(dataloader)
    losses.append(epoch_loss)
    
    scheduler.step(epoch_loss)

    if epoch == 0 or (epoch+ 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Plotting the losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.show()

name_of_model = 'retina_segmentation_model_full.pth'
torch.save(model.state_dict(), name_of_model)