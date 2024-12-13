import numpy as np
import os


class Parameters:
    EEG_FREQ = 128
    EEG_TIME_WINDOW = 4
    EEG_SHAPE = (16, EEG_FREQ * EEG_TIME_WINDOW)

    #PPG_FREQ = 32
    PPG_FREQ = 128
    PPG_TIME_WINDOW = 6
    PPG_SHAPE = (1, PPG_FREQ * PPG_TIME_WINDOW)

    #EDA_FREQ = 32
    EDA_FREQ = 64
    EDA_TIME_WINDOW = 4
    EDA_SHAPE = (1, EDA_FREQ * EDA_TIME_WINDOW)

    RESP_FREQ = 32
    RESP_TIME_WINDOW = 10
    RESP_SHAPE = (1, RESP_FREQ * RESP_TIME_WINDOW)


def is_valid(d):
    d_eeg = d['eeg']
    d_ppg = d['ppg']
    d_eda = d['eda']
    d_resp = d['resp']

    valid = True

    if (d_eeg is None or d_ppg is None
            or d_eda is None or d_resp is None):
        valid = False

    if d_eeg is not None and d_eeg.shape != Parameters.EEG_SHAPE:
        raise Exception(f'EEG data has shape {d_eeg.shape}, '
                        f'which is not the desired shape {Parameters.EEG_SHAPE}.')

    if d_ppg is not None and d_ppg.shape != Parameters.PPG_SHAPE:
        raise Exception(f'PPG data has shape {d_ppg.shape}, '
                        f'which is not the desired shape {Parameters.PPG_SHAPE}.')

    if d_eda is not None and d_eda.shape != Parameters.EDA_SHAPE:
        raise Exception(f'EDA data has shape {d_eda.shape}, '
                        f'which is not the desired shape {Parameters.EDA_SHAPE}.')

    if d_resp is not None and d_resp.shape != Parameters.RESP_SHAPE:
        raise Exception(f'RESP data has shape {d_resp.shape}, '
                        f'which is not the desired shape {Parameters.RESP_SHAPE}.')

    return valid


def main():
    data = np.load(os.path.join(os.getcwd(), 'data', 'data.npy'), allow_pickle=True)
    print(f'Number of all samples: {len(data)}')

    data_with_all_modalities_list = []

    for d in data:
        if is_valid(d):
            data_with_all_modalities_list.append(d)

    data_with_all_modalities = np.array(data_with_all_modalities_list, dtype=[
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

    print(f'Number of samples that have all modalities: {len(data_with_all_modalities)}')
    np.save(os.path.join(os.getcwd(), 'data', 'data_with_all_modalities.npy'), data_with_all_modalities)


if __name__ == "__main__":
    main()
