import sys
sys.path.append('ssynth-release-main/code/data_generation/')

import util
import random
import pandas as pd

def get_df():
    l_model = util.get_l_model()
    l_hairModel = util.get_l_hairModel()
    l_lesion = util.get_l_lesion()
    l_times = [10, 20, 30, 40]
    l_lesionMat = range(len(util.get_l_lesionMat()))
    l_melanosomes = util.get_l_melanosomes()
    l_light = util.get_l_light()
    l_hairAlbedoIndex = util.get_l_hairAlbedoIndex()
    l_fractionBlood = util.get_l_fractionBlood()

    l_param_list = []

    # generate lesions
    lesion_params = []
    for i in range(10):
      this_lesion = {
        'id_model': random.choice(l_model),
        'id_hairModel': random.choice(l_hairModel),
        'id_lesion': random.choice(l_lesion),
        'id_timePoint': random.choice(l_times),
        'id_lesionMat': random.choice(l_lesionMat),
        'id_mel': random.choice(l_melanosomes),
        'id_hairAlbedo': random.choice(l_hairAlbedoIndex),
        'id_fracBlood': random.choice(l_fractionBlood),
        'offset': -2,
        'origin_y': 15.0,
        'mi_variant': 'cuda_spectral',
        'lesion_scale': 1.5
      }
      lesion_params.append(this_lesion)

    # for each lesion, generate 5 mel fractions and 5 light conditions
    mel_fractions = [0.01, 0.11, 0.31, 0.41]
    light_conds = [8]

    render_params = []
    for params in lesion_params:
      for mel in mel_fractions:
        for light in light_conds:
          this_params = params.copy()
          this_params['id_mel'] = mel
          this_params['id_light'] = light
          render_params.append(this_params)

    df = pd.DataFrame(render_params)

    print(df.head())
    print(len(df))
    return df

if __name__ == "__main__":
    df = get_df()
    print(df.head())
    #id_model,id_hairModel,id_lesion,id_timePoint,id_lesionMat,id_fracBlood,id_mel,id_light,id_hairAlbedo,offset,origin_y,mi_variant,lesion_scale

    df.to_csv('data/supporting_data/params_lists/mel_lighting_variation.csv', index=False)