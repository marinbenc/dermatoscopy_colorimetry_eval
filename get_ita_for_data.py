import os
import sys
import shutil
import glob
import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import concurrent.futures
from kmeans_skin_color_estimator import find_dominant_color, get_ita_angle, get_fitzpatrick_type
import seaborn as sns
import pandas as pd
import cv2
from PIL import Image
import math
from skimage import io, color
from matplotlib import pyplot as plt
import numpy as np

## --- CALCULATION METHODS --- ##


# FROM S-SYNTH code -- "GT" ITA calculation because it uses the mask
# although this still isn't perfectly correlated with melanosome fraction
def get_ita_value(img_path, mask_path):
    '''
    calculate ita and color from image/mask pair
    
    Parameters: 
        - img_path: path for an image
        - mask_path: path for the corresponding ground truth mask
        
    Returns: 
        - ITA: estimated ITA score for the skin
        - color: estimated color of the skin based the ITA score
    '''

    def calculate_mean_cannel_value(channel_array, mask):
        c_mean_temp = np.median(channel_array[mask == 0], axis=0)
        c_std = np.std(channel_array[mask == 0], axis=0)
        c_mean = np.median(
            channel_array[(channel_array >= c_mean_temp - c_std) & (channel_array <= c_mean_temp + c_std)], axis=0)
        return c_mean

    def calculate_ITA(L, b):
        ITA = np.arctan((L - 50) / b) * (180 / np.pi)
        return ITA

    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[:, :, 1]
    mask = (mask > 0).astype(int)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L_channel = np.multiply(lab[:, :, 0], (mask == 0))
    b_channel = np.multiply(lab[:, :, 2], (mask == 0))
    L = calculate_mean_cannel_value(L_channel, mask)
    b = calculate_mean_cannel_value(b_channel, mask)
    ITA = calculate_ITA(L, b)

    if ITA <= 10:
        color = "dark"
    elif 10 < ITA <= 19:
        color = "tan1"
    elif 19 < ITA <= 28:
        color = "tan2"
    elif 28 < ITA <= 34.5:
        color = "int1"
    elif 34.5 < ITA <= 41:
        color = "int2"
    elif 41 < ITA <= 48:
        color = "lt1"
    elif 48 < ITA <= 55:
        color = "lt2"
    elif ITA >= 55:
        color = "very_lt"
    return ITA, color

def calculate_ita_ssynth(file_paths):
  itas = np.zeros(len(file_paths))
  mask_paths = [f.replace('image.png', 'mask.png') for f in file_paths]
  with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(get_ita_value, file_paths, mask_paths))
    for i, (ita, _) in enumerate(results):
        itas[i] = ita

  return itas

# From Bencevic et al. 2024
def _process_image_kmeans(file_path):
    img = cv.imread(file_path)
    # resize
    img = cv.resize(img, (128, 128))
    if 'dermis' in file_path:
        img[-80:, -50:] = 0 # remove logo at bottom right corner
    label = None
    try:
      dominant_color = find_dominant_color(img, label).squeeze()
      ita_angle = get_ita_angle(dominant_color)
      fp_type = get_fitzpatrick_type(ita_angle)
    except Exception as e:
      print('Error processing file: ', e)
      return np.array((-1, -1, -1)), -99, -1
    return dominant_color, ita_angle, fp_type

def calculate_ita_kmeans(file_paths):
  colors = np.zeros((len(file_paths), 3))
  itas = np.zeros(len(file_paths))
  skin_types = np.zeros(len(file_paths))

  with concurrent.futures.ProcessPoolExecutor() as executor:
      results = list(executor.map(_process_image_kmeans, file_paths))
      for i, (dominant_color, ita_angle, fp_type) in enumerate(results):
          colors[i] = dominant_color
          itas[i] = ita_angle
          skin_types[i] = fp_type

  return itas

## From Bevan et Atapour-Abarghouei
## from https://github.com/tkalbl/RevisitingSkinToneFairness/blob/main/BevanCorrection.ipynb
def bin_ITA(ita_max):
    ita_bnd_kin=-1
    if ita_max > 55:
        ita_bnd_kin = 1
    if 41 < ita_max <= 55:
        ita_bnd_kin = 2
    if 28 < ita_max <= 41:
        ita_bnd_kin = 3
    if 19 < ita_max <= 28:
        ita_bnd_kin = 4
    if 10 < ita_max <= 19:
        ita_bnd_kin = 5
    if ita_max <= 10:
        ita_bnd_kin = 6
    return ita_bnd_kin

# Hair removal for ITA calculation
def hair_remove(image):
    # Convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Kernel for morphologyEx
    kernel = cv2.getStructuringElement(1, (17, 17))
    # Apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Apply thresholding to blackhat
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # Inpaint with original image and threshold image
    final_image = cv2.inpaint(image, threshold, 1, cv2.INPAINT_TELEA)
    return final_image

# Calculates Fitzpatrick skin type of an image using Kinyanjui et al.'s thresholds
def get_sample_ita_kin(path,size=256):
    ita_type = -1
    ita2_type = -1
    try:
        im = cv2.imread(path) # cv2 image is numpy array
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # if image wider than long centre crop by smallest side
        if im.shape[1] < im.shape[0]:
            dim1 = int(im.shape[1])
            dim0 = int(im.shape[0])
            chi = int((dim0 - dim1) / 2)
            im = im[chi:chi + dim1, 0:dim1]
        else:
            # if image longer than wide centre crop by smallest side
            dim1 = int(im.shape[0])
            dim0 = int(im.shape[1])
            chi = int((dim0 - dim1) / 2)
            im = im[0:dim1, chi:chi + dim1]
        im = Image.fromarray(im, 'RGB') # converts to PIL image for resizing
        # resizing
        im = im.resize((size, size), Image.Resampling.LANCZOS)
        #rgb = io.imread(path) # no longer needed
        rgb = np.array(im) # skimage requires a np array again
        # blur
        rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
        #rgb = hair_remove(rgb) # disabled hair removal because it does not work well for dark skin
        lab = color.rgb2lab(rgb)

        ita_lst = []
        ita2_lst = []
        ita_bnd_lst = []

        # Taking samples from different parts of the image
        L1 = lab[230:250, 115:135, 0].mean()
        b1 = lab[230:250, 115:135, 2].mean()

        L2 = lab[5:25, 115:135, 0].mean()
        b2 = lab[5:25, 115:135, 2].mean()

        L3 = lab[115:135, 5:25, 0].mean()
        b3 = lab[115:135, 5:25, 2].mean()

        L4 = lab[115:135, 230:250, 0].mean()
        b4 = lab[115:135, 230:250, 2].mean()

        L5 = lab[216:236, 216:236, 0].mean()
        b5 = lab[216:236, 216:236, 2].mean()

        L6 = lab[216:236, 20:40, 0].mean()
        b6 = lab[216:236, 20:40, 2].mean()

        L7 = lab[20:40, 20:40, 0].mean()
        b7 = lab[20:40, 20:40, 2].mean()

        L8 = lab[20:40, 216:236, 0].mean()
        b8 = lab[20:40, 216:236, 2].mean()

        L_lst = [L1, L2, L3, L4, L5, L6, L7, L8]
        b_lst = [b1, b2, b3, b4, b5, b6, b7, b8]

        # Calculating ITA values
        for L, b in zip(L_lst, b_lst):
            ###### Own contribution start
            ita2 = math.atan2((L - 50), b) * (180 / math.pi)
            ita2_lst.append(ita2)
            ###### Own contribution end
            ita = math.atan((L - 50) / b) * (180 / math.pi)
            ita_lst.append(ita)

        # Using max ITA value (lightest)
        ita_max = max(ita_lst)
        ita2_max =  max(ita2_lst)
        # Getting skin shade band from ITA
        ita_type = bin_ITA(ita_max)
        ita2_type = bin_ITA(ita2_max)
    except Exception:
        pass

    return ita_max,ita2_max

def calculate_ita_bevan(file_paths):
  itas = np.zeros(len(file_paths))
  with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(get_sample_ita_kin, file_paths))
    for i, (_, ita2) in enumerate(results):
        itas[i] = ita2

  return itas

if __name__ == "__main__":
  # find all image.pngs
  all_files = glob.glob('data/output_10k/output/**/image.png', recursive=True)

  # find a list of mel values
  mel_values = []
  for f in all_files:
      mel = f.split('/')[5]
      mel = float(mel.split('_')[1])
      mel_values.append(mel)

  itas_ssynth = calculate_ita_ssynth(all_files)
  itas_kmeans = calculate_ita_kmeans(all_files)
  itas_bevan = calculate_ita_bevan(all_files)

  df = pd.DataFrame({
      'file': all_files,
      'mel': mel_values,
      'ita_ssynth': itas_ssynth,
      'ita_kmeans': itas_kmeans,
      'ita_bevan': itas_bevan,
  })

  df.to_csv('ita_values.csv', index=False)