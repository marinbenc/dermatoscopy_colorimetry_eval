from get_ita_for_data import bin_ITA
import pandas as pd
import numpy as np

df = pd.read_csv('ita_values.csv')
df[df['ita_kmeans'] == -99] = np.nan
df = df.dropna()

df['light'] = df['file'].apply(lambda x: x.split('/')[-3])
df['mel'] = df['mel'].astype(float)

df['ssynth_fp'] = df['ita_ssynth'].apply(bin_ITA).astype('category')

# get mel range per ssynth_fp
mean_mel = df.groupby('ssynth_fp')['mel'].mean()
std_mel = df.groupby('ssynth_fp')['mel'].std()
mel_range = pd.concat([mean_mel, std_mel], axis=1)
mel_range.columns = ['mean', 'std']
# get mid points between each category
mel_range['mid'] = mel_range['mean'].rolling(2).mean()
mel_range = mel_range.dropna()
display(mel_range)

def bin_FP_by_mel(mel):
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

mel_range.to_csv('mel_range.csv')
