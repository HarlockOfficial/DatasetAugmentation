import random
import sys

import numpy as np
import tensorflow as tf
from moabb.datasets import PhysionetMI

from DatasetAugmentation import utils

random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)
np.random.RandomState(42)


def main(out_class, path_to_neural_network):
    if out_class.lower() == "none":
        out_class = None

    generator, is_generator_new = utils.eeg_generator(out_class, path_to_neural_network, return_is_new=True, is_training=False)
    assert not is_generator_new, "Please train the generator first"

    x, y = utils.load_dataset(PhysionetMI)
    if out_class is not None:
        dataset_by_label = utils.split_dataset_by_label(x, y)
        testing_dataset = dataset_by_label[out_class]
    else:
        testing_dataset = x
    dataset_element = random.choice(testing_dataset)
    avg_element = np.mean(testing_dataset, axis=0)
    seed = tf.random.normal([22, 58, 65])
    # score the generator
    generated_element = generator(seed, training=False)['output_0']
    avg_generated_element = np.mean(generated_element, axis=0)
    element_diff = np.subtract(avg_generated_element, avg_element)
    print("Dataset Element:", dataset_element)
    print("Generated Element:", generated_element)
    print("Element Difference:", element_diff)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: Tester.py <network_expected_output_class> <path_to_network_files_to_load>")
        sys.exit(1)
    expected_output_class = sys.argv[1]
    path_to_network_files = sys.argv[2]
    main(expected_output_class, path_to_network_files)
