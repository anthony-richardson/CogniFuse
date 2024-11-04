import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    data = np.load(os.path.join(os.getcwd(), 'data', 'data_with_all_modalities.npy'), allow_pickle=True)

    ids = [int(d['participant_id']) for d in data]

    bins = np.arange(0, 170, 1)  # fixed bin size
    plt.xlim([min(ids) - 1, max(ids) + 1])
    plt.hist(ids, bins=bins, alpha=0.5)
    plt.title('Samples per participant')
    plt.xlabel('participant id')
    plt.ylabel('count')

    plt.savefig(os.path.join(os.getcwd(), 'preprocessing', 'samples_per_participant.png'))


if __name__ == "__main__":
    main()
