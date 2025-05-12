import numpy as np
import torch
import os
from torch.utils.data import DataLoader


class CogniFitDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tasks=None):
        # We decide to load all data instead of loading on demand. Depending on the usa case
        # data loaders should only provide a subset of the data (e.g. for one task), requiring
        # to search through all samples.
        complete_data = np.load(data_path, allow_pickle=True)
        filtered_data = []
        # Filtering
        for d in complete_data:
            if tasks is None or d['scenario'] in tasks:
                filtered_data.append((
                    d['participant_id'],
                    d['scenario'],
                    d['eeg'],
                    d['ppg'],
                    d['eda'],
                    d['resp'],
                ))

        self.data = np.array(filtered_data, dtype=[
                ('participant_id', 'i4'),
                ('scenario', 'U30'),
                ('eeg', 'O'),
                ('ppg', 'O'),
                ('eda', 'O'),
                ('resp', 'O')
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        modality_data = {}
        for key in ['eeg', 'ppg', 'eda', 'resp']:
            modality_data[key] = entry[key]

        meta_info = {}

        for key in ['participant_id', 'scenario']:
            meta_info[key] = entry[key]

        return modality_data, meta_info


def get_data_loader(batch_size, tasks, data_dir, split, shuffle=True, drop_last=False):
    split_file = split + '.npy'
    data_path = os.path.join(data_dir, split_file)
    dataset = CogniFitDataset(data_path=data_path, tasks=tasks)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, drop_last=drop_last
    )
    return loader



