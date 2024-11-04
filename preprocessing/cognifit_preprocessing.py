import numpy as np
import os
import mne

import matplotlib.pyplot as plt

ADC_EEG = 'eeg.adc'
ADC_GLOVE = 'glove.adc'
ADC_RESP = 'respiration.adc'
TIMESTAMPS = '.timestamps.txt'

TIMESTAMP_EEG = ADC_EEG + TIMESTAMPS
TIMESTAMP_GLOVE = ADC_GLOVE + TIMESTAMPS
TIMESTAMP_RESP = ADC_RESP + TIMESTAMPS

TIME_WINDOW_SHIFT = 2


class Parameters:
    EEG_REC_FREQ = 256
    PPG_REC_FREQ = 256
    EDA_REC_FREQ = 256
    RESP_REC_FREQ = 512

    EEG_LOW_FILTER = 0.1
    EEG_HIGH_FILTER = 40
    EEG_FREQ = 128
    EEG_TIME_WINDOW = 4

    PPG_LOW_FILTER = 0.5
    PPG_HIGH_FILTER = 20
    PPG_FREQ = 32
    PPG_TIME_WINDOW = 6

    EDA_LOW_FILTER = 0.01
    EDA_HIGH_FILTER = 4
    EDA_FREQ = 32
    EDA_TIME_WINDOW = 4

    RESP_LOW_FILTER = 0.05
    RESP_HIGH_FILTER = 1
    RESP_FREQ = 32
    RESP_TIME_WINDOW = 10


def preprocess(sensor_name, data, ts):
    rec_freq = getattr(Parameters, f'{sensor_name}_REC_FREQ')
    low_filter = getattr(Parameters, f'{sensor_name}_LOW_FILTER')
    high_filter = getattr(Parameters, f'{sensor_name}_HIGH_FILTER')
    freq = getattr(Parameters, f'{sensor_name}_FREQ')
    time_window = getattr(Parameters, f'{sensor_name}_TIME_WINDOW')

    '''if sensor_name == 'EDA':
        # EEG: 4*256
        # PPG: 6*256
        # EDA: 4*256
        # RESP: 10*512

        end_step = 1000+(4*256)

        plt.plot(data[0, 1000:end_step])
        plt.show()

        # Filtering'
        data = mne.filter.filter_data(data, sfreq=rec_freq, l_freq=low_filter, h_freq=high_filter)

        plt.plot(data[0, 1000:end_step])
        plt.show()

        #exit()'''

    # Filtering'
    data = mne.filter.filter_data(data, sfreq=rec_freq, l_freq=low_filter, h_freq=high_filter)

    # Splitting
    data_segments = []
    segment_mask = []
    start_time = ts[0][1]
    while start_time + time_window <= ts[-1][1]:
        end_time = start_time + time_window

        start_index = int(min(ts, key=lambda r: np.linalg.norm(r[1] - start_time))[0])
        end_index = int(min(ts, key=lambda r: np.linalg.norm(r[1] - end_time))[0])

        segment = data[:, start_index:end_index]

        # For ignoring segments with less than 90% of expected data points. These occur mostly at the end
        # because the time stamps are recorded longer than the actual sensor data.
        if segment.shape[1] > time_window * rec_freq * 0.9:
            num_missing_frames = time_window * rec_freq - segment.shape[1]

            if num_missing_frames > 0:
                # Take beginning frames from next segment
                next_frames = data[:, end_index:(end_index + num_missing_frames)]
                if next_frames.shape[1] != num_missing_frames:
                    # Special case: Not enough frames left at the end
                    segment_mask.append(0)
                else:
                    segment = np.append(segment, next_frames, axis=1)
                    segment_mask.append(1)
            # Too many frames
            elif num_missing_frames < 0:
                # Remove from beginning
                segment = segment[:, -num_missing_frames:]
                segment_mask.append(1)
            # Exactly the desired number of frames
            else:
                segment_mask.append(1)
        else:
            segment_mask.append(0)

        data_segments.append(segment)
        start_time += TIME_WINDOW_SHIFT

    # Down sampling
    down_sampling_factor = rec_freq/freq
    data_segments = [mne.filter.resample(s, down=down_sampling_factor, npad='auto') for s in data_segments]

    '''if sensor_name == 'RESP':
        for s in data_segments:
            print(s.shape)
            s = np.transpose(s, (1, 0))
            print(s.shape)

            print(s)

            plt.plot(s)
            plt.show()

        exit()'''

    #if sensor_name == 'EEG':
    #for s in data_segments:
    #    print(s.shape)

    #print(segment_mask)

    #exit()

    #print(len(data_segments))
    #data_segments = [data_segments[i] for i in xrange(len(lst)) if msk[i]] # data_segments[segment_mask]
    #print(len(data_segments))

    #exit()
    return data_segments, segment_mask


def get_cutting_indices(root_path):
    available_ts = {}
    cutting_indices = {}

    # Load text files with timestamps
    try:
        ts_eeg = np.loadtxt(os.path.join(root_path, TIMESTAMP_EEG))
        if len(ts_eeg.shape) == 2:
            available_ts['eeg'] = ts_eeg
    except FileNotFoundError:
        pass
    try:
        ts_glove = np.loadtxt(os.path.join(root_path, TIMESTAMP_GLOVE))
        if len(ts_glove.shape) == 2:
            available_ts['glove'] = ts_glove
    except FileNotFoundError:
        pass
    try:
        ts_resp = np.loadtxt(os.path.join(root_path, TIMESTAMP_RESP))
        if len(ts_resp.shape) == 2:
            available_ts['resp'] = ts_resp
    except FileNotFoundError:
        pass

    if len(available_ts) != 0:
        max_first_ts = max([e[0][1] for e in available_ts.values()])
        min_last_ts = min([e[-1][1] for e in available_ts.values()])

        for name, ts in available_ts.items():
            first_index = int(min(ts, key=lambda r: np.linalg.norm(r[1] - max_first_ts))[0])
            last_index = int(min(ts, key=lambda r: np.linalg.norm(r[1] - min_last_ts))[0])
            cutting_indices[name] = (first_index, last_index)

    return cutting_indices, available_ts


'''def apply_cut(list_to_trim, end_index):
    return list_to_trim[:end_index] if list_to_trim is not None else None

def apply_mask():
    return'''


def get_segment(segments, index, segments_mask, sensor_name):
    freq = getattr(Parameters, f'{sensor_name}_FREQ')
    time_window = getattr(Parameters, f'{sensor_name}_TIME_WINDOW')

    try:
        d = segments[index]
        if segments_mask[index] == 0:
            d = None
        elif d.shape[1] != freq * time_window:
            raise Exception('Something wnt wrong. Not the desired amount of frames.')
    # Index error if there not enough segments.
    # Type error if segments is None.
    except (IndexError, TypeError) as e:
        d = None

    return d


def add_to_data_list(data_list, participant_id, task_name, task_specific_dirs):
    for d in task_specific_dirs:

        difficulty = -1
        file_name_parts = d.split('/')[-1].split('_')
        if len(file_name_parts) > 1 and file_name_parts[-2].isnumeric():
            difficulty = int(file_name_parts[-2])
            if difficulty == 0:
                raise Exception('The difficulty should never be 0 as this '
                                'number is reserved for the trial number')

        dirs = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))

        for directory in dirs:

            '''data_segments_eeg = None
            segment_mask_eeg = None

            data_segments_ppg = None
            segment_mask_ppg = None
            data_segments_eda = None
            segment_mask_eda = None

            data_segments_resp = None
            segment_mask_resp = None'''

            print(os.path.join(d, directory))

            cutting_indices, available_ts = get_cutting_indices(os.path.join(d, directory))

            # Data loading
            try:
                eeg_file_path = os.path.join(d, directory, ADC_EEG)
                file_eeg = open(eeg_file_path, 'r')
                data_eeg = np.fromfile(file_eeg, np.int16).reshape((-1, 16)).T.astype(np.float64)
                try:
                    eeg_start_index, eeg_end_index = cutting_indices['eeg']
                    data_eeg = data_eeg[:, eeg_start_index:eeg_end_index]
                    data_segments_eeg, segment_mask_eeg = preprocess('EEG', data_eeg, available_ts['eeg'])
                except KeyError:
                    # No timestamps
                    data_segments_eeg = None
                    segment_mask_eeg = None
            except FileNotFoundError:
                # No data
                data_segments_eeg = None
                segment_mask_eeg = None

            try:
                glove_file_path = os.path.join(d, directory, ADC_GLOVE)
                file_glove = open(glove_file_path, 'r')
                data_glove = np.fromfile(file_glove, np.int16).reshape((-1, 2)).T.astype(np.float64)
                try:
                    glove_start_index, glove_end_index = cutting_indices['glove']
                    data_glove = data_glove[:, glove_start_index:glove_end_index]
                    data_ppg = np.expand_dims(data_glove[0, :], axis=0)
                    data_segments_ppg, segment_mask_ppg = preprocess('PPG', data_ppg, available_ts['glove'])
                    data_eda = np.expand_dims(data_glove[1, :], axis=0)
                    data_segments_eda, segment_mask_eda = preprocess('EDA', data_eda, available_ts['glove'])
                except KeyError:
                    data_segments_ppg = None
                    segment_mask_ppg = None
                    data_segments_eda = None
                    segment_mask_eda = None
            except FileNotFoundError:
                data_segments_ppg = None
                segment_mask_ppg = None
                data_segments_eda = None
                segment_mask_eda = None

            try:
                resp_file_path = os.path.join(d, directory, ADC_RESP)
                file_resp = open(resp_file_path, 'r')
                data_resp = np.fromfile(file_resp, np.int16).T.astype(np.float64)
                data_resp = np.expand_dims(data_resp, axis=0)
                try:
                    resp_start_index, resp_end_index = cutting_indices['resp']
                    data_resp = data_resp[:, resp_start_index:resp_end_index]
                    data_segments_resp, segment_mask_resp = (
                        preprocess('RESP', data_resp, available_ts['resp']))
                except KeyError:
                    data_segments_resp = None
                    segment_mask_resp = None
            except FileNotFoundError:
                data_segments_resp = None
                segment_mask_resp = None

            '''min_nr_segments = min(len(data_segments_eeg) if data_segments_eeg is not None else np.inf,
                                  len(data_segments_ppg) if data_segments_ppg is not None else np.inf,
                                  len(data_segments_eda) if data_segments_eda is not None else np.inf,
                                  len(data_segments_resp) if data_segments_resp is not None else np.inf)

            # Cutting the last segments where not all modalities are present
            data_segments_eeg = apply_cut(data_segments_eeg, min_nr_segments)
            data_segments_ppg = apply_cut(data_segments_ppg, min_nr_segments)
            data_segments_eda = apply_cut(data_segments_eda, min_nr_segments)
            data_segments_resp = apply_cut(data_segments_resp, min_nr_segments)

            segment_mask_eeg = apply_cut(segment_mask_eeg, min_nr_segments)
            segment_mask_ppg = apply_cut(segment_mask_ppg, min_nr_segments)
            segment_mask_eda = apply_cut(segment_mask_eda, min_nr_segments)
            segment_mask_resp = apply_cut(segment_mask_resp, min_nr_segments)

            segment_masks = [segment_mask_eeg, segment_mask_ppg, segment_mask_eda, segment_mask_resp]
            segment_masks = [np.array(m) for m in segment_masks if m is not None]

            # TODO: use segment masks to calculate OR and then filter out the joint segments

            #print(segment_mask_eeg, segment_mask_ppg, segment_mask_eda) #, segment_mask_resp)
            #print(segment_mask_eeg and segment_mask_ppg and segment_mask_eda)#, segment_mask_resp))
            #print(np.logical_and(np.array(segment_mask_eeg), np.array(segment_mask_ppg), np.array(segment_mask_eda)))
            print(np.logical_and(*segment_masks))
    
            segment_masks_all_modalities = np.logical_and(*segment_masks)
            print(len(segment_masks_all_modalities))
            print(len(data_segments_eeg))'''

            max_nr_segments = max(len(data_segments_eeg) if data_segments_eeg is not None else 0,
                                  len(data_segments_ppg) if data_segments_ppg is not None else 0,
                                  len(data_segments_eda) if data_segments_eda is not None else 0,
                                  len(data_segments_resp) if data_segments_resp is not None else 0)
            #print(max_nr_segments)

            #print(len(data_segments_eeg), len(data_segments_ppg), len(data_segments_eda))

            for i in range(max_nr_segments):
                eeg_segment = get_segment(data_segments_eeg, i, segment_mask_eeg, 'EEG')
                ppg_segment = get_segment(data_segments_ppg, i, segment_mask_ppg, 'PPG')
                eda_segment = get_segment(data_segments_eda, i, segment_mask_eda, 'EDA')
                resp_segment = get_segment(data_segments_resp, i, segment_mask_resp, 'RESP')

                #segment_list = [eeg_segment, ppg_segment, eda_segment, resp_segment]

                if (eeg_segment is not None or ppg_segment is not None or
                        eda_segment is not None or resp_segment is not None):
                    data_list.append((
                        participant_id,
                        task_name,
                        difficulty,
                        eeg_segment,
                        ppg_segment,
                        eda_segment,
                        resp_segment
                    ))

            #print(len(data_list))


            #exit()



            '''for i in range(min_nr_segments):
                data_list.append((
                    participant_id,
                    task_name,
                    difficulty,
                    data_segments_eeg[i] if data_segments_eeg is not None else None,
                    data_segments_ppg[i] if data_segments_ppg is not None else None,
                    data_segments_eda[i] if data_segments_eda is not None else None,
                    data_segments_resp[i] if data_segments_resp is not None else None
                ))'''


def main():
    cognifit_dir = os.path.join(os.getcwd(), '/share/data/Cognifit')

    numbers = [str(n).zfill(3) for n in range(170)]
    data_dirs = [d for d in os.listdir(cognifit_dir) if d in numbers]

    task_names = [
        'Relax_before_LCT',         # pause before LCT task (controlled, no LCT)
        'Relax_during_LCT',         # pause in between LCT task (controlled, no LCT)
        'Relax_after_LCT',          # pause after LCT task (controlled, no LCT)
        'SwitchingTask',         # Switching paradigm (controlled, no LCT)
        'LCT_Baseline',         # LCT without additional task (uncontrolled, LCT)
        'SwitchBackAuditive',   # LCT with added auditive task (uncontrolled, LCT + auditive)
        'VisualSearchTask'      # LCT with added visual task (uncontrolled, LCT + visual)
    ]

    data_list = []

    for participant_id in data_dirs:
        participant_dirs = os.listdir(os.path.join(cognifit_dir, participant_id))
        participant_dirs = list(filter(lambda x: 'Train' not in x, participant_dirs))

        for task_name in task_names:
            task_specific_dirs = list(filter(lambda x: x.startswith(task_name), participant_dirs))
            # Sorting so that all repetitions are iteratively removed
            task_specific_dirs.sort()

            for d in task_specific_dirs:
                parts = d.split('_')
                # Ignoring recordings that were repeated.
                # Assuming that the last number is always the trial number and that all files
                # track such a trial number. Generally should be 0 except in case of retries.
                if parts[-1].isnumeric():  # and parts[-2].isnumeric():
                    if d[:-1] + str(int(d[-1]) + 1) in task_specific_dirs:
                        task_specific_dirs.remove(d)
                else:
                    raise Exception(f'Directory {d} does not have a trial number.')

            task_specific_dirs = [os.path.join(cognifit_dir, participant_id, d) for d in task_specific_dirs]
            add_to_data_list(data_list, int(participant_id), task_name, task_specific_dirs)

    data = np.array(data_list, dtype=[
        ('participant_id', 'i4'),
        ('task', 'U30'),
        ('difficulty', 'i4'),
        ('eeg', 'O'),
        ('ppg', 'O'),
        ('eda', 'O'),
        ('resp', 'O')
    ]
    )

    np.save(os.path.join(os.getcwd(), 'data', 'data.npy'), data)


if __name__ == "__main__":
    main()
