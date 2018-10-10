from torch.utils.data import Dataset
import pandas as pd
from train_config import data_path_config
import numpy as np
import os


# labels_dict = {-2: 0,
#                -1: 1,
#                0: 2,
#                1: 3}

content_labels_dict = {-1: 0,
                       0: 1,
                       1: 2}

labels_array = np.array([-2, -1, 0, 1])
labels_dict = {k: v for v, k in enumerate(labels_array)}


def calculate_labels(predictions):
    label_idx = np.argmax(predictions, axis=1)
    return labels_array[label_idx]


def save_predictions(predictions, path, target):
    predict_labels = calculate_labels(predictions)
    if os.path.exists(path):
        data = pd.read_csv(path)
    else:
        data = pd.DataFrame()
    data[target] = predict_labels
    data.to_csv(path)
    return


class UserCommentDataset(Dataset):
    """
    User Comment Dataset
    """

    def __init__(self, path, targets=None, content='content', transformers=None):
        full_content = pd.read_csv(path)
        self.content = full_content[content].values
        self.transformers = transformers
        if not targets:
            self.targets = None
            self.is_train = False
            return
        if isinstance(targets, str):
            targets = [targets]
        self.targets = full_content[targets].values
        self.is_train = True

    def __len__(self):
        return self.content.shape[0]

    def __getitem__(self, idx):
        content = self.content[idx].lstrip('"').rstrip('"')
        if self.transformers:
            for transform in self.transformers:
                content = transform(content)

        if self.is_train:
            targets = self.targets[idx, :]
            targets_list = []
            for target in targets:
                target_cat = np.zeros(len(labels_array), dtype='int8')
                target_cat[labels_dict[target]] = 1
                targets_list.append(target_cat)

        else:
            targets_list = None
        return content, targets_list


if __name__ == '__main__':
    data_path = data_path_config['train_data_path']
    # train_location = UserCommentDataset(data_path, 'location_traffic_convenience')
    # for i in range(len(train_location)):
    #     content, target = train_location[i]
    #     print(content)
    #     print(target)
    #     if i == 10:
    #         break

    train_has_location = UserCommentDataset(data_path, 'location_traffic_convenience')
    for i in range(len(train_has_location)):
        content, target = train_has_location[i]
        print(content)
        print(target)
        if i == 50:
            break
