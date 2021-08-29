# System Libs.
import random

# Other Libs
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class MaskClassifierDataset(Dataset):

    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(list(img_paths))

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        label = torch.tensor(self.labels[index])

        if self.transform:
            image = self.transform(image)

        return (image, label)


def train_test_split_df(data_df, crit_col='path', test_size=0.2, shuffle=True, random_state=None):
    if random_state:
        random.seed(random_state)

    _idxs = set(data_df[crit_col].unique())
    _size = len(_idxs)
    _size_test = int(_size * test_size)

    # Split DataFrame
    test_idxs = set(random.sample(_idxs, _size_test))
    test_df = data_df.loc[data_df[crit_col].isin(test_idxs)]

    train_idxs = _idxs - test_idxs
    train_df = data_df.loc[data_df[crit_col].isin(train_idxs)]

    return (train_df, test_df)


def dist_analysis(df_dict, col_list=['Mask', 'Age', 'Gender']):
    dist_list = []
    for name, df in df_dict.items():
        dist_df_list = []
        for col in col_list:
            # Dist. info
            dist_count = pd.DataFrame(df[col].value_counts())
            dist_count.columns = ['Count']
            dist_ratio = pd.DataFrame(df[col].value_counts(True))
            dist_ratio.columns = ['Ratio']

            # Construct & append dataframe
            _dist_df = pd.concat([dist_count, dist_ratio], axis=1)
            _dist_df.index = pd.MultiIndex.from_product(
                [[col], _dist_df.index])
            dist_df_list.append(_dist_df)
        dist_df = pd.concat(dist_df_list, axis=0)
        dist_df.columns = pd.MultiIndex.from_product([[name], dist_df.columns])
        dist_list.append(dist_df)
    dist_info = pd.concat(dist_list, axis=1)

    return dist_info
