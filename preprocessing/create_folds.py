import numpy as np
import os
import random

'''
    There are 134 participants with at least one session where all modalities are recorded.
    We choose to randomly, but fixed, separate 10% of the participants for testing.
    For the remaining participants perform 10-fold cross validation.
    The resulting splits are 108:12:14 in participants or 81:9:10 in percent.
    We fix and store these folds so that all models may use the same splits.
'''


def create_array(data_list):
    arr = np.array(data_list, dtype=[
            ('participant_id', 'i4'),
            #('task', 'U30'),
            #('difficulty', 'i4'),
            ('scenario', 'U30'),
            ('eeg', 'O'),
            ('ppg', 'O'),
            ('eda', 'O'),
            ('resp', 'O')
        ]
    )
    return arr


def main():
    data = np.load(os.path.join(os.getcwd(), 'data', 'data_with_all_modalities.npy'), allow_pickle=True)
    print(f'Total number of samples: {len(data)}')

    ids = [int(d['participant_id']) for d in data]
    unique_ids = list(set(ids))

    # Fixed seed so that these folds are reproducible.
    random.Random(420).shuffle(unique_ids)

    folds_dir = os.path.join(os.getcwd(), 'data', 'folds')
    os.mkdir(folds_dir)

    test_ids = unique_ids[:14]
    test_data = [d for d in data if int(d['participant_id']) in test_ids]
    test_data = create_array(test_data)
    print(f'Shared test: {len(test_data)}')
    np.save(os.path.join(os.getcwd(), 'data', 'folds', 'test.npy'), test_data)

    unique_ids = unique_ids[14:]

    for i in range(10):
        fold_i_dir = os.path.join(folds_dir, f'{i+1}')
        os.mkdir(fold_i_dir)

        validation_ids = unique_ids[i:i+12]
        train_ids = [x for x in unique_ids if x not in validation_ids]

        validation_data = [d for d in data if int(d['participant_id']) in validation_ids]
        train_data = [d for d in data if int(d['participant_id']) in train_ids]

        validation_data = create_array(validation_data)
        train_data = create_array(train_data)

        np.save(os.path.join(fold_i_dir, 'validation.npy'), validation_data)
        np.save(os.path.join(fold_i_dir, 'train.npy'), train_data)

        print(f'Fold {i+1}: {len(validation_data)} (validation), {len(train_data)} (training)')


if __name__ == "__main__":
    main()
