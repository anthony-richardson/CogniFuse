import numpy as np
import os
#import mne
import random
from scipy.signal import butter, filtfilt, resample
from tqdm import tqdm

ADC_EEG = 'eeg.adc'
ADC_GLOVE = 'glove.adc'
ADC_RESP = 'respiration.adc'
TIMESTAMPS = '.timestamps.txt'

TIMESTAMP_EEG = ADC_EEG + TIMESTAMPS
TIMESTAMP_GLOVE = ADC_GLOVE + TIMESTAMPS
TIMESTAMP_RESP = ADC_RESP + TIMESTAMPS

TIME_WINDOW_SHIFT = 2


class Parameters:
    EEG_FILTER_ORDER = 4
    EEG_FILTER_LOW_CUT = 0.1
    EEG_FILTER_HIGH_CUT = 40
    EEG_REC_FREQ = 256
    EEG_FREQ = 128
    EEG_TIME_WINDOW = 4

    # Even though we filter out high signal frequencies,
    # we might still want a high sampling frequency to allow
    # precice detection of heart beats for the calculation of
    # inter-beat intervalls and to avoid aliasing. (Source: ChatGPT)
    # Lapitan et al. found that reducing the upper cutoff frequency
    # of band-pass filtering below 10 Hz leads to damping of the dicrotic
    # notch and a phase shift of the pulse wave signal. We aoid this as it
    # might be useful for ognitive load detection.
    #PPG_LOW_FILTER = 0.5
    #PPG_HIGH_FILTER = 20
    #PPG_LOW_FILTER = None
    #PPG_HIGH_FILTER = 5.5
    PPG_FILTER_ORDER = 2
    PPG_FILTER_LOW_CUT = 0.01
    PPG_FILTER_HIGH_CUT = 10
    #PPG_FREQ = 32
    PPG_REC_FREQ = 256
    PPG_FREQ = 128
    PPG_TIME_WINDOW = 6

    # Fast chages in EDA, which relate to cognitive load, might be better
    # detected when maintaining a higher sampling rate. (Source: ChatGPT)
    #EDA_LOW_FILTER = 0.01
    #EDA_HIGH_FILTER = 4
    EDA_FILTER_ORDER = 2
    EDA_FILTER_LOW_CUT = 0.01
    EDA_FILTER_HIGH_CUT = 1
    #EDA_FREQ = 32
    EDA_REC_FREQ = 256
    EDA_FREQ = 64
    EDA_TIME_WINDOW = 4

    RESP_FILTER_ORDER = 2
    RESP_FILTER_LOW_CUT = 0.05
    RESP_FILTER_HIGH_CUT = 1
    #RESP_LOW_FILTER = None
    #RESP_HIGH_FILTER = 0.45
    #RESP_LOW_FILTER = None
    #RESP_HIGH_FILTER = 1
    RESP_REC_FREQ = 512
    RESP_FREQ = 32
    RESP_TIME_WINDOW = 10


def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Forward-backward filter to avoid phase shift.
    y = filtfilt(b, a, data)
    return y


def split_segments(data, ts, rec_freq, time_window):
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

    return data_segments, segment_mask


def preprocess(sensor_name, data, ts):
    ####
    # To plot the signals before and after preprocessing. 
    # For the full preprocessing, this needs to be set to False. 
    visualize_and_exit = False
    # The modality to plot
    sensor_to_visualize = "RESP"
    ####

    filter_order = getattr(Parameters, f'{sensor_name}_FILTER_ORDER')
    filter_low_cut = getattr(Parameters, f'{sensor_name}_FILTER_LOW_CUT')
    filter_high_cut = getattr(Parameters, f'{sensor_name}_FILTER_HIGH_CUT')
    rec_freq = getattr(Parameters, f'{sensor_name}_REC_FREQ')
    freq = getattr(Parameters, f'{sensor_name}_FREQ')
    time_window = getattr(Parameters, f'{sensor_name}_TIME_WINDOW')

    num_samples_before_downsampling  = time_window * rec_freq
    num_samples_after_downsampling  = time_window * freq

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

    #filter_len = int((1 / min(max(high_filter * 0.25, 2.), rec_freq / 2. - high_filter)) * 3.3 * rec_freq)



    #filter_len = (1 / high_filter) * 3.3
    # Filtering'
    """data = mne.filter.filter_data(
        data,
        sfreq=rec_freq,
        l_freq=low_filter,
        h_freq=high_filter,
        #filter_length=filter_len
    )"""

    if visualize_and_exit:
        original_data_segments, _ = split_segments(data, ts, rec_freq, time_window)
        original_data_segments = [
            resample(s, num=num_samples_after_downsampling, axis=-1) 
            if s.shape[-1] == num_samples_before_downsampling else s for s in original_data_segments
        ]


    data = butter_bandpass_filtfilt(
        data, 
        lowcut=filter_low_cut, 
        highcut=filter_high_cut, 
        fs=rec_freq, 
        order=filter_order
    )

    # Splitting
    data_segments, segment_mask = split_segments(data, ts, rec_freq, time_window)

    """data_segments = []
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
        start_time += TIME_WINDOW_SHIFT"""

    #down_sampling_factor = rec_freq/freq
    #data_segments = [mne.filter.resample(s, down=down_sampling_factor, npad='auto') for s in data_segments]

    """if sensor_name == "RESP":
        for s in data_segments:
            print(s.shape)"""
    
    # Only down sampling the samples that should be used in the end
    data_segments = [
        resample(s, num=num_samples_after_downsampling, axis=-1) 
        if s.shape[-1] == num_samples_before_downsampling else s for s in data_segments
    ]

    """if sensor_name == "RESP":
        for s in data_segments:
            print(s.shape)

        print(sensor_name)

        #exit()"""

    #print(sensor_name)


    if visualize_and_exit: 
        # Inspecting filter results
        if sensor_name == sensor_to_visualize:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for i, (original_s, s) in enumerate(zip(original_data_segments, data_segments)):
                original_s = np.transpose(original_s, (1, 0))
                s = np.transpose(s, (1, 0))
                
                if sensor_name == "EEG":
                    plt.rcParams["figure.figsize"] = (30,30)
                    for c in range(s.shape[-1]):
                        plt.subplot(16, 2, c*2 + 1)
                        if c == 0:
                            plt.title("Before filtering")
                        plt.plot(original_s[:, c])

                        plt.subplot(16, 2, c*2 + 2)
                        if c == 0:
                            plt.title("After filtering")
                        plt.plot(s[:, c])
                else:
                    plt.rcParams["figure.figsize"] = (15,10)

                    plt.subplot(1, 2, 1)
                    plt.title("Before filtering")
                    plt.plot(original_s)
                    
                    plt.subplot(1, 2, 2)
                    plt.title("After filtering")
                    plt.plot(s)

                plt.savefig(os.path.join(os.getcwd(), "preprocessing", 
                            "filter_examples", f"{sensor_name}", f"{i}.png"))
                plt.clf()
            exit()

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


def add_to_data_list(data_list, new_participant_id, task_name, task_specific_dirs):
    for d in task_specific_dirs:

        difficulty = -1
        file_name_parts = d.split('/')[-1].split('_')
        if len(file_name_parts) > 1 and file_name_parts[-2].isnumeric():
            difficulty = int(file_name_parts[-2])
            if difficulty == 0:
                raise Exception('The difficulty should never be 0 as this '
                                'number is reserved for the trial number')

        scenario = task_name
        if difficulty > 0:
            scenario += f'_{difficulty}'

        dirs = list(filter(lambda x: os.path.isdir(os.path.join(d, x)), os.listdir(d)))

        for directory in dirs:
            #print(os.path.join(d, directory))

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

            max_nr_segments = max(len(data_segments_eeg) if data_segments_eeg is not None else 0,
                                  len(data_segments_ppg) if data_segments_ppg is not None else 0,
                                  len(data_segments_eda) if data_segments_eda is not None else 0,
                                  len(data_segments_resp) if data_segments_resp is not None else 0)

            for i in range(max_nr_segments):
                eeg_segment = get_segment(data_segments_eeg, i, segment_mask_eeg, 'EEG')
                ppg_segment = get_segment(data_segments_ppg, i, segment_mask_ppg, 'PPG')
                eda_segment = get_segment(data_segments_eda, i, segment_mask_eda, 'EDA')
                resp_segment = get_segment(data_segments_resp, i, segment_mask_resp, 'RESP')

                if (eeg_segment is not None or ppg_segment is not None or
                        eda_segment is not None or resp_segment is not None):
                    data_list.append((
                        new_participant_id,
                        scenario,
                        #task_name,
                        #difficulty,
                        eeg_segment,
                        ppg_segment,
                        eda_segment,
                        resp_segment
                    ))


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

    # Generating new participant ids with fixed random seed
    # for internal reproducability. This script must not be made public.
    new_id_dict = {}
    random.seed(4253)
    for participant_id in data_dirs:
        new_id = random.randint(1000000, 9999999)
        while new_id in new_id_dict.values():
            new_id = random.randint(1000000, 9999999)
        new_id_dict[participant_id] = new_id

    for participant_id in tqdm(data_dirs):
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
                if parts[-1].isnumeric():
                    if d[:-1] + str(int(d[-1]) + 1) in task_specific_dirs:
                        task_specific_dirs.remove(d)
                else:
                    raise Exception(f'Directory {d} does not have a trial number.')

            task_specific_dirs = [os.path.join(cognifit_dir, participant_id, d) for d in task_specific_dirs]

            new_participant_id = new_id_dict[participant_id]
            #add_to_data_list(data_list, int(participant_id), task_name, task_specific_dirs)
            add_to_data_list(data_list, new_participant_id, task_name, task_specific_dirs)

    data = np.array(data_list, dtype=[
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

    np.save(os.path.join(os.getcwd(), 'data', 'data.npy'), data)


if __name__ == "__main__":
    main()
