import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms

class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = int(self.df.iloc[idx]['emotion'])
        pixels_str = self.df.iloc[idx]['pixels']
        pixels = np.fromstring(pixels_str, sep=' ', dtype=np.uint8)
        image = pixels.reshape(48, 48)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(csv_path, batch_size=64):
    """ფუნქცია, რომელიც ავტომატურად ყოფს მონაცემებს და აბრუნებს DataLoader-ებს"""
    full_df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(
        full_df,
        test_size=0.2,
        random_state=42,
        stratify=full_df['emotion']
    )

    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = FERDataset(dataframe=train_df, transform=base_transform)
    val_dataset = FERDataset(dataframe=val_df, transform=base_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
