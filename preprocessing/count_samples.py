import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    data = np.load(os.path.join(os.getcwd(), 'data', 'data_with_all_modalities.npy'), allow_pickle=True)

    print(len(data))


if __name__ == "__main__":
    main()
