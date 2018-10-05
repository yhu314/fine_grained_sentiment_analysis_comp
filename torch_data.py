from torch.utils.data import Dataset
import pandas as pd
from train_config import data_path_config
from keras.utils import to_categorical
import numpy as np
import os


labels_dict = {-2: 0,
               -1: 1,
               0: 1,
               1: 2}

content_labels_dict = {-1: 0,
                       0: 1,
                       1: 2}

labels_array = np.array([-2, -1, 0, 1])


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

    def __init__(self, data_path, target, binary=True, content='content'):
        full_content = pd.read_csv(data_path)
        self.content = full_content[content].values
        self.binary = binary
        self.labels_dict = labels_dict
        self.target = np.ones(full_content.shape[0])
        if target:
            self.target = full_content[target].values

    def __len__(self):
        return self.content.shape[0]

    def __getitem__(self, idx):
        target = self.target[idx]
        content = self.content[idx]
        if isinstance(content_labels_dict, list):
            content = [x for x in content if x != '']
        if self.binary:
            target = int(target != -2)
            target = to_categorical(target, num_classes=2)
        else:
            target = to_categorical(labels_dict[target], num_classes=len(self.labels_dict))
        return content, target


class UserCommentContainDataset(UserCommentDataset):
    def __init__(self,  data_path, target, content='content'):
        full_content = pd.read_csv(data_path)
        has_comment = (full_content[target] != -2)
        self.content = full_content.loc[has_comment, content].values
        self.target = full_content.loc[has_comment, target].values
        self.binary = False # In this dataset, we always need to return the true tag
        self.labels_dict = content_labels_dict


if __name__ == '__main__':
    data_path = data_path_config['train_data_path']
    # train_location = UserCommentDataset(data_path, 'location_traffic_convenience')
    # for i in range(len(train_location)):
    #     content, target = train_location[i]
    #     print(content)
    #     print(target)
    #     if i == 10:
    #         break

    train_has_location = UserCommentContainDataset(data_path, 'location_traffic_convenience')
    for i in range(len(train_has_location)):
        content, target = train_has_location[i]
        print(content)
        print(target)
        if i == 50:
            break
