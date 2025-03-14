import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2
import os
import glob
import re
import torchvision.models as models
from coral_pytorch.layers import CoralLayer
from coral_pytorch.dataset import levels_from_labelbatch
from coral_pytorch.losses import coral_loss
import matplotlib.pyplot as plt

def bin_FP_by_mel(mel_range, mel):
  # find the category that mel falls into
  # if mel is less than the first mid point, return the first category
  if mel < mel_range['mid'].iloc[0]:
    return 1
  # if mel is greater than the last mid point, return the last category
  if mel > mel_range['mid'].iloc[-1]:
    return mel_range.index[-1]
  # find the category that mel falls into
  for i in range(len(mel_range) - 1):
    if mel_range['mid'].iloc[i] <= mel < mel_range['mid'].iloc[i + 1]:
      return mel_range.index[i]
  return None

class FPDataset(Dataset):
    def __init__(self, root_dir, files, transform=None):
        # search for image.png recursively
        self.root_dir = root_dir
        self.transform = transform
        self.files = glob.glob(os.path.join(root_dir, '**', 'image.png'), recursive=True)

        # only include files in `files`
        if files is not None:
            self.files = [f for f in self.files if f in files]

        self.files = sorted(self.files)

        # mel_range.csv is generated in analysis.ipynb
        mel_range = pd.read_csv('mel_range.csv', index_col=0)

        # search for e.g. /mel_0.21/ in the path
        self.mels = []
        for f in self.files:
            m = re.search(r'mel_([0-9]+\.[0-9]+)', f)
            self.mels.append(float(m.group(1)))
        self.mels = np.array(self.mels)

        self.fps = [bin_FP_by_mel(mel_range, mel) for mel in self.mels]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        
        fp = self.fps[idx] - 1
        fp = torch.tensor(fp, dtype=torch.long)
        return img, fp

model = models.vgg11_bn(pretrained=True)
model.classifier[6] = CoralLayer(4096, 6)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.GaussianBlur(21, sigma=5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == '__main__':
    # train / valid / test split
    all_dataset = FPDataset('data/output_10k', files=None, transform=transform)
    all_files = all_dataset.files
    all_files = np.sort(np.array(all_files))
    np.random.seed(0)
    np.random.shuffle(all_files)
    train_files = all_files[:int(0.8*len(all_files))]
    valid_files = all_files[int(0.8*len(all_files)):int(0.9*len(all_files))]
    test_files = all_files[int(0.9*len(all_files)):]

    # save the split to csv
    pd.DataFrame(train_files).to_csv('data/train_files.csv', index=False)
    pd.DataFrame(valid_files).to_csv('data/valid_files.csv', index=False)
    pd.DataFrame(test_files).to_csv('data/test_files.csv', index=False)

    train_dataset = FPDataset('data/output_10k', train_files, transform=transform)
    valid_dataset = FPDataset('data/output_10k', valid_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Early stopping
    best_valid_loss = float('inf')
    patience = 5
    counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        for i, (x, y) in enumerate(train_loader):
            levels = levels_from_labelbatch(y, num_classes=6)
            levels = levels.to(device)
            # plt.imshow(x[0].permute(1, 2, 0))
            # plt.show()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)

            loss = coral_loss(logits, levels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):
                levels = levels_from_labelbatch(y, num_classes=6)
                levels = levels.to(device)
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = coral_loss(logits, levels)
                valid_loss += loss.item()
            valid_loss /= len(valid_loader)

        print(f'Epoch {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'mel_model.pth')
            counter = 0
        else:
            counter += 1
            if counter == patience:
                break