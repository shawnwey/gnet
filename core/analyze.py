import ntpath
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as M
from scipy.special import expit
from sklearn.metrics import log_loss
from tqdm.notebook import tqdm

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def analyze_net_pklResults(exp_id: str):
    exp_folder = Path(os.path.join('output_test', exp_id))
    net_name = exp_id.split('-')[0]
    column_list = ['video', 'score', 'label']

    # Load data in multi-index dataframe
    data_model_list = []
    model_list = []
    # for model_folder in tqdm(results_model_folder):
    data_dataset_list = []
    dataset_list = []
    pkl_results = exp_folder.glob('*.pkl')
    for pkl_path in pkl_results:
        pkl_name = os.path.splitext(ntpath.split(pkl_path)[1])[0]
        df_frames = pd.read_pickle(pkl_path)[column_list]
        # Add info on training and test datasets
        data_dataset_list.append(df_frames)
        dataset_list.append(pkl_name)
    data_model_list.append(pd.concat(data_dataset_list, keys=dataset_list, names=['dataset']))
    model_list.append(exp_id)
    data_frame_df = pd.concat(data_model_list, keys=model_list, names=['model']).swaplevel(0, 1)

    # per-frame test
    net_list = [net_name]
    # 排列组合
    comb_list = list(combinations(net_list, 1))
    iterables = [dataset_list, ['loss', 'auc', 'acc']]
    index = pd.MultiIndex.from_product(iterables, names=['dataset', 'metric'])
    results_df = pd.DataFrame(index=index, columns=comb_list)

    for dataset in dataset_list:
        print(dataset)
        for model_comb in tqdm(comb_list):
            df = get_df(data_frame_df, dataset)
            results_df[model_comb][dataset, 'loss'] = log_loss(df['label'], expit(np.mean(df[list(model_comb)],
                                                                                          axis=1)))
            results_df[model_comb][dataset, 'auc'] = M.roc_auc_score(df['label'], expit(np.mean(df[list(model_comb)],
                                                                                                axis=1)))
            results_df[model_comb][dataset, 'acc'] = M.accuracy_score(df['label'], expit(np.mean(df[list(model_comb)],
                                                                                                axis=1)))
    print(results_df.T)


def get_df(video_all_df, dataset):
    selected_df = video_all_df.loc[dataset].unstack(['model'])['score']
    exp_ids = selected_df.columns
    aux_df = video_all_df.loc[dataset].unstack(['model'])['video']
    selected_df['video'] = aux_df[aux_df.columns[0]]
    selected_df['label'] = video_all_df.loc[dataset].unstack(['model'])['label'].mean(axis=1)
    mapper = dict()
    for exp_id in exp_ids:
        mapper[exp_id] = exp_id.split('-')[0]
    selected_df = selected_df.rename(mapper, axis=1)
    return selected_df


if __name__ == '__main__':
    analyze_net_pklResults('EfficientNetB4-baseline')
