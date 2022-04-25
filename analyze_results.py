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


def analyze_one_net():
    exp_id = ''
    model_name = 'EfficientNetB4'
    ckpt_name = 'bestval.pth'

    results_model_path = Path('output_test/EfficientNetB4')
    result_model = os.path.join('output', 'model_name', 'weights', ckpt_name)
    column_list = ['video', 'score', 'label']

    # Load data in multi-index dataframe
    if os.path.exists('data_frame_df.pkl'):
        data_frame_df = pd.read_pickle('data_frame_df.pkl')
        model_results = results_model_path.glob('*.pkl')
        dataset_list = []
        for model_path in model_results:
            dataset_tag = os.path.splitext(ntpath.split(model_path)[1])[0]
            dataset_list.append(dataset_tag)
    else:
        data_dataset_list = []
        dataset_list = []
        # train_model_tag = model_folder.name
        model_results = results_model_path.glob('*.pkl')
        for model_path in model_results:
            traindb = 'ff-c40-720-140-140'

            testdb, testsplit = model_path.with_suffix('').name.rsplit('_',1)
            dataset_tag = os.path.splitext(ntpath.split(model_path)[1])[0]
            df_frames = pd.read_pickle(model_path)[column_list]
            # Add info on training and test datasets
            df_frames['netname'] = model_name
            df_frames['train_db'] = traindb
            df_frames['test_db'] = testdb
            df_frames['test_split'] = testsplit
            data_dataset_list.append(df_frames)
            dataset_list.append(dataset_tag)
        # data_model_list.append(pd.concat(data_dataset_list, keys=dataset_list, names=['dataset']))
        # model_list.append(train_model_tag)
        # data_frame_df = pd.concat(data_model_list, keys=model_list, names=['model']).swaplevel(0, 1)

        # data_frame_df.to_pickle('data_frame_df.pkl')
    pass


def analyze():
    results_root = Path('output_test/')
    results_model_folder = list(results_root.glob('*'))
    column_list = ['video', 'score', 'label']
    do_distplot = False

    # Load data in multi-index dataframe
    if os.path.exists('data_frame_df.pkl'):
        data_frame_df = pd.read_pickle('data_frame_df.pkl')
        model_list = []
        for model_folder in tqdm(results_model_folder):
            dataset_list = []
            train_model_tag = model_folder.name
            model_results = model_folder.glob('*.pkl')
            for model_path in model_results:
                dataset_tag = os.path.splitext(ntpath.split(model_path)[1])[0]
                dataset_list.append(dataset_tag)
            model_list.append(train_model_tag)
    else:
        data_model_list = []
        model_list = []
        for model_folder in tqdm(results_model_folder):
            data_dataset_list = []
            dataset_list = []
            train_model_tag = model_folder.name
            model_results = model_folder.glob('*.pkl')
            for model_path in model_results:
                netname = 'EfficientNetB4'
                traindb = 'ff-c40-720-140-140'
                # netname = train_model_tag.split('net-')[1].split('_')[0]
                # traindb = train_model_tag.split('traindb-')[1].split('_')[0]
                testdb, testsplit = model_path.with_suffix('').name.rsplit('_',1)
                dataset_tag = os.path.splitext(ntpath.split(model_path)[1])[0]
                df_frames = pd.read_pickle(model_path)[column_list]
                # Add info on training and test datasets
                df_frames['netname'] = netname
                df_frames['train_db'] = traindb
                df_frames['test_db'] = testdb
                df_frames['test_split'] = testsplit
                data_dataset_list.append(df_frames)
                dataset_list.append(dataset_tag)
            data_model_list.append(pd.concat(data_dataset_list, keys=dataset_list, names=['dataset']))
            model_list.append(train_model_tag)
        data_frame_df = pd.concat(data_model_list, keys=model_list, names=['model']).swaplevel(0, 1)
        data_frame_df.to_pickle('data_frame_df.pkl')

    # per-frame test
    net_list = list(data_frame_df['netname'].unique())
    # 排列组合
    comb_list_1 = list(combinations(net_list, 1))
    comb_list_2 = list(combinations(net_list, 2))
    comb_list_3 = list(combinations(net_list, 3))
    comb_list_4 = list(combinations(net_list, 4))
    comb_list = comb_list_1 + comb_list_2 + comb_list_3 + comb_list_4
    iterables = [dataset_list, ['loss', 'auc']]
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

    print(results_df.T)


def get_df(video_all_df, dataset):
    selected_df = video_all_df.loc[dataset].unstack(['model'])['score']
    models = selected_df.columns
    aux_df = video_all_df.loc[dataset].unstack(['model'])['video']
    selected_df['video'] = aux_df[aux_df.columns[0]]
    selected_df['label'] = video_all_df.loc[dataset].unstack(['model'])['label'].mean(axis=1)
    mapper = dict()
    for model in models:
        mapper[model] = model.split('net-')[1].split('_traindb')[0]
    selected_df = selected_df.rename(mapper, axis=1)
    return selected_df


if __name__ == '__main__':
    analyze()
