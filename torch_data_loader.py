import numpy as np
import torch
from torch.utils.data import Dataset

import settings


class IMBDDataset(Dataset):
    """
    IMBD sentiment dataset in PyTorch format;
    inherits torch.utils.data.Dataset
    """

    def __init__(self, data_list, target_list):
        """
        @param data_list: list of review tokens
        @param target_list: list of review targets
        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, key):
        """
        Triggered when calling dataset[i]
        """
        token_idx = self.data_list[key][:settings.CONFIG["max_sent_len"]]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]


def review_collate_func(batch):
    """
    Custom DataLoader() function;
    dynamically pads batch data
    """
    data_list = []
    label_list = []
    length_list = []
    # print("collate batch: ", batch[0][0])
    # batch[0][0] = batch[0][0][:settings.CONFIG["max_sent_len"]]

    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])

    # Padding
    for datum in batch:
        padded_vec = np.pad(
            np.array(datum[0]),
            pad_width=(
                (
                    0,
                    settings.CONFIG["max_sent_len"] - datum[1]
                )),
            mode="constant",
            constant_values=0,
        )
        data_list.append(padded_vec)
    return [
        torch.from_numpy(np.array(data_list)),
        torch.LongTensor(length_list),
        torch.LongTensor(label_list),
    ]


def get(idx, targets, shuffle):
    """
    Helper function
    """
    dataset = IMBDDataset(idx, targets)
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=settings.CONFIG["batch_size"],
        collate_fn=review_collate_func,
        shuffle=shuffle,
    )
    return loader
