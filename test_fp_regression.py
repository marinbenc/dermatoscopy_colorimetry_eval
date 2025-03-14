import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import glob
import re
from torch.utils.data import Dataset

from coral_pytorch.dataset import corn_label_from_logits
from train_fp_regression import FPDataset, model, transform

class ISICDataset(Dataset):
  # images from data/isic_2020 (no labels, only images)
  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.transform = transform
    self.files = glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True)
    self.files = sorted(self.files)[:1000] # only use the first 100 images

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    img = cv2.imread(self.files[idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = self.transform(img)
    return img

def test_mel_regression():
    test_files = pd.read_csv('data/test_files.csv')['0'].values
    test_dataset = FPDataset('data/output_10k', files=test_files, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_model = model.to('cuda')
    test_model.load_state_dict(torch.load('mel_model.pth'))
    test_model.eval()

    # fps = []
    # preds = []
    # for i, (img, fp) in enumerate(test_loader):
    #     img = img.to('cuda')
    #     fp = fp.to('cuda')
    #     pred = test_model(img)
    #     pred = corn_label_from_logits(pred)
    #     fps.append(fp.item() + 1)
    #     preds.append(pred.item() + 1)

    # fps = np.array(fps)
    # preds = np.array(preds)

    # df = pd.DataFrame({'fp': fps, 'pred': preds, 'file': test_dataset.files})
    # print(df.head())
    # df.to_csv('fp_regression_test_results.csv', index=False)

    # test on isic
    isic_dataset = ISICDataset('data/isic_2020', transform)
    isic_loader = torch.utils.data.DataLoader(isic_dataset, batch_size=1, shuffle=False)

    isic_preds = []
    for i, img in enumerate(isic_loader):
        img = img.to('cuda')
        pred = test_model(img)
        pred = corn_label_from_logits(pred)
        isic_preds.append(pred.item() + 1)

    isic_preds = np.array(isic_preds)
    df = pd.DataFrame({'pred': isic_preds, 'file': isic_dataset.files})
    df.to_csv('isic_fp_regression_test_results.csv', index=False)

if __name__ == '__main__':
  test_mel_regression()