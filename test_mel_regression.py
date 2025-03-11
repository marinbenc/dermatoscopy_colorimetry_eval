import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2

from train_mel_regression import MelDataset, model, transform

def test_mel_regression():
    test_files = pd.read_csv('data/test_files.csv')['0'].values
    test_dataset = MelDataset('data/output_10k', files=test_files, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_model = model.to('cuda')
    test_model.load_state_dict(torch.load('mel_model.pth'))
    test_model.eval()

    mels = []
    preds = []
    for i, (img, mel) in enumerate(test_loader):
        img = img.to('cuda')
        mel = mel.to('cuda')
        pred = test_model(img)
        mels.append(mel.item())
        preds.append(pred.item())

    mels = np.array(mels)
    preds = np.array(preds)

    # compare mels and preds
    df = pd.DataFrame({'mel': mels, 'pred': preds, 'file': test_dataset.files})
    print(df.head())
    df.to_csv('mel_regression_test_results.csv', index=False)

if __name__ == '__main__':
  test_mel_regression()
