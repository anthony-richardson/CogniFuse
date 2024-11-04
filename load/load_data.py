import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from scipy import signal

import matplotlib.pyplot as plt


class CogniFitDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, tasks=None, difficulties=None):
        # We decide to load all data instead of loading on demand. Depending on the usa case
        # data loaders should only provide a subset of the data (e.g. for one task), requiring
        # to search through all samples.
        complete_data = np.load(data_path, allow_pickle=True)
        filtered_data = []
        # Filtering
        for d in complete_data:
            #scenario = d['task']
            if d['difficulty'] > 0:
                # d['difficulty'] -= 1
                d['task'] += f'_{d['difficulty']}'

            if ((tasks is None or d['task'] in tasks) and
                    (difficulties is None or d['difficulty'] in difficulties)):

                # Enumerating tasks
                #d['task'] = task_enumerator[d['task']]

                # Changing to zero based labels (originally 1, 2, 3)
                '''scenario = d['task']
                if d['difficulty'] > 0:
                    #d['difficulty'] -= 1
                    scenario += f"_{d['difficulty']}"'''

                #filtered_data.append(d)
                filtered_data.append((
                    d['participant_id'],
                    d['task'],
                    d['eeg'],
                    d['ppg'],
                    d['eda'],
                    d['resp'],
                ))

        # TODO: change preprocessing of data so that it is already structured as a scenario
        #   Then remove the logic above accordingly.
        self.data = np.array(filtered_data, dtype=[
                ('participant_id', 'i4'),
                ('scenario', 'U30'),
                #('difficulty', 'i4'),
                ('eeg', 'O'),
                ('ppg', 'O'),
                ('eda', 'O'),
                ('resp', 'O')
            ]
        )

        #print(set(self.data['difficulty']))

        #for e in set(self.data['task']):
        #    print(e)

        #exit()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        modality_data = {}
        for key in ['eeg', 'ppg', 'eda', 'resp']:
            modality_data[key] = entry[key]

        meta_info = {}

        #for key in ['participant_id', 'task', 'difficulty']:
        for key in ['participant_id', 'scenario']:
            meta_info[key] = entry[key]

        return modality_data, meta_info


'''task_enumerator = {
    'Relax_before_LCT': 0,          # pause before LCT task (controlled, no LCT)
    'Relax_during_LCT': 1,          # pause in between LCT task (controlled, no LCT)
    'Relax_after_LCT': 2,           # pause after LCT task (controlled, no LCT)
    'SwitchingTask': 3,             # Switching paradigm (controlled, no LCT)
    'LCT_Baseline': 4,              # LCT without additional task (uncontrolled, LCT)
    'SwitchBackAuditive': 5,        # LCT with added auditive task (uncontrolled, LCT + auditive)
    'VisualSearchTask': 6           # LCT with added visual task (uncontrolled, LCT + visual)
}'''


def get_data_loader(batch_size, tasks, data_dir, split, shuffle=True):
    split_file = split + '.npy'
    data_path = os.path.join(data_dir, split_file)
    dataset = CogniFitDataset(data_path=data_path, tasks=tasks)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=8, drop_last=False
    )
    return loader



