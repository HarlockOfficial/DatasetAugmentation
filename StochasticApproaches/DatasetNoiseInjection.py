import datetime
import os
import pickle

import numpy as np
from moabb.datasets import PhysionetMI

import DatasetAugmentation.utils

np.random.seed(42)

class NoiseInjection(object):
    def __init__(self, shape: tuple=None, dataset: np.ndarray=None):
        if shape is None:
            shape = (58, 65)
        assert dataset.shape[1:] == shape
        self.shape = shape
        # take only second and third quartile of the dataset, to avoid going out of bounds
        self.dataset = self.__filter_dataset(dataset, NoiseInjection.__take_second_and_third_quartile)

    def __compute_entry(self):
        entry_index = np.random.choice(self.dataset.shape[0])
        entry = self.dataset[entry_index]
        noise = np.random.normal(0, 1, entry.shape)
        return entry + noise

    def __call__(self, *args, **kwargs):
        entry = self.__compute_entry()
        if 'shape' in kwargs:
                return np.reshape(entry, kwargs['shape'])
        return self.__compute_entry()

    @staticmethod
    def __take_second_and_third_quartile(dataset):
        dataset = np.sort(dataset, axis=0)
        quartile = dataset.shape[0] // 4
        return dataset[quartile: 3 * quartile]

    def __filter_dataset(self, dataset, function):
        assert dataset.shape[1:] == self.shape
        filtered = function(dataset)
        return filtered


def main():
    x, y = DatasetAugmentation.utils.load_dataset(PhysionetMI)
    dataset_by_class = DatasetAugmentation.utils.split_dataset_by_label(x, y)
    t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    for key in dataset_by_class.keys():
        noise_injected_dataset = NoiseInjection(dataset=np.array(dataset_by_class[key]), shape=(58, 65))
        x = noise_injected_dataset()
        print(x)
        print(x.shape)
        if not os.path.exists(f'./models/{t}_noise_injector/'):
            os.makedirs(f'./models/{t}_noise_injector/')
        with open(f'./models/{t}_noise_injector/generator_{key}.pkl', 'wb') as f:
            pickle.dump(noise_injected_dataset, f)


if __name__ == '__main__':
    main()
