import datetime
import os
import pickle

import numpy as np
from moabb.datasets import PhysionetMI

import DatasetAugmentation.utils

np.random.seed(42)

class RandomSampler(object):
    def __init__(self, min_entry: np.ndarray = None, max_entry: np.ndarray = None, shape: tuple=None, dataset: np.ndarray=None):
        if shape is None:
            shape = (58, 65)
        self.shape = shape
        if min_entry is None and max_entry is None:
            self.min_entry = self.__filter_dataset(dataset, lambda x: np.min(x, axis=0))
            self.max_entry = self.__filter_dataset(dataset, lambda x: np.max(x, axis=0))
        else:
            assert min_entry.shape == max_entry.shape == self.shape
            self.min_entry = min_entry
            self.max_entry = max_entry

    def __compute_entry(self):
        return np.random.uniform(self.min_entry, self.max_entry, size=self.shape)

    def __call__(self, *args, **kwargs):
        entry = self.__compute_entry()
        if 'shape' in kwargs:
                return np.reshape(entry, kwargs['shape'])
        return self.__compute_entry()

    def __filter_dataset(self, dataset, function):
        assert dataset.shape[1:] == self.shape
        filtered = function(dataset)
        return filtered


def main():
    x, y = DatasetAugmentation.utils.load_dataset(PhysionetMI)
    dataset_by_class = DatasetAugmentation.utils.split_dataset_by_label(x, y)
    t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    for key in dataset_by_class.keys():
        random_sampler = RandomSampler(dataset=np.array(dataset_by_class[key]), shape=(58, 65))
        x = random_sampler()
        print(x)
        print(x.shape)
        if not os.path.exists(f'./models/{t}_random_sampler/'):
            os.makedirs(f'./models/{t}_random_sampler/')
        with open(f'./models/{t}_random_sampler/generator_{key}.pkl', 'wb') as f:
            pickle.dump(random_sampler, f)


if __name__ == '__main__':
    main()
