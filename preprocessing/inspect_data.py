import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    data = np.load(os.path.join(os.getcwd(), 'data', 'data_with_all_modalities.npy'), allow_pickle=True)

    for d in data:
        for m in ['eeg', 'ppg', 'eda', 'resp']:
            d_m = d[m]
            print(m)

            # For eeg only first channel and others only have one
            plt.plot(d_m[0])
            plt.show()


if __name__ == "__main__":
    main()
